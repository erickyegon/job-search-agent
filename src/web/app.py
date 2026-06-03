"""FastAPI web app for job search agent SaaS."""

import os
import json
import logging
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from ..database import UserDB, PLAN_LIMITS
from ..resume_parser import parse_resume

logger = logging.getLogger(__name__)

app = FastAPI(title="JobMatch AI", description="AI-powered job search agent")

TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Pricing (cents for Stripe)
PRICING = {
    "free": {"name": "Free", "price": 0, "price_display": "$0", "period": "forever"},
    "pro": {"name": "Pro", "price": 999, "price_display": "$9.99", "period": "/month"},
    "enterprise": {"name": "Enterprise", "price": 2999, "price_display": "$29.99", "period": "/month"},
}


def get_db() -> UserDB:
    return UserDB(os.environ.get("DB_PATH", "jobs.db"))


# ---- Pages ----

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {
        "request": request,
        "pricing": PRICING,
        "plan_limits": PLAN_LIMITS,
    })


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get("/dashboard/{user_id}", response_class=HTMLResponse)
async def dashboard(request: Request, user_id: int):
    db = get_db()
    try:
        user = db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        jobs = db.get_user_jobs(user_id, min_score=60, limit=50)
        profile = db.get_user_profile(user_id)
        limits = db.get_plan_limits(user_id)
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": user,
            "jobs": jobs,
            "profile": profile,
            "limits": limits,
            "plan_info": PRICING.get(user["plan"], PRICING["free"]),
        })
    finally:
        db.close()


# ---- API Endpoints ----

@app.post("/api/signup")
async def api_signup(
    email: str = Form(...),
    name: str = Form(""),
    telegram_chat_id: str = Form(""),
    resume: UploadFile = File(None),
    discount_code: str = Form(""),
):
    db = get_db()
    try:
        # Check if email already exists
        existing = db.get_user_by_email(email)
        if existing:
            return JSONResponse(
                {"error": "Email already registered", "user_id": existing["id"]},
                status_code=409,
            )

        # Create user
        user = db.create_user(email=email, name=name, telegram_chat_id=telegram_chat_id)
        user_id = user["id"]

        # Parse resume if uploaded
        if resume and resume.filename:
            try:
                file_bytes = await resume.read()
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                result = await parse_resume(resume.filename, file_bytes, api_key)
                db.update_profile(user_id, result["profile"], result["resume_text"])
            except Exception as e:
                logger.warning(f"Resume parse failed for user {user_id}: {e}")
                # User still created, just no profile yet

        # Apply discount code
        discount_result = None
        if discount_code:
            discount_result = db.apply_discount(user_id, discount_code)

        return JSONResponse({
            "success": True,
            "user_id": user_id,
            "message": "Account created! Check your email for job matches.",
            "discount": discount_result,
            "dashboard_url": f"/dashboard/{user_id}",
        })
    finally:
        db.close()


@app.post("/api/upload-resume/{user_id}")
async def api_upload_resume(user_id: int, resume: UploadFile = File(...)):
    db = get_db()
    try:
        user = db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        file_bytes = await resume.read()
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        result = await parse_resume(resume.filename, file_bytes, api_key)
        db.update_profile(user_id, result["profile"], result["resume_text"])

        return JSONResponse({
            "success": True,
            "profile": result["profile"],
            "message": "Resume parsed and profile updated!",
        })
    finally:
        db.close()


@app.get("/api/jobs/{user_id}")
async def api_get_jobs(user_id: int, min_score: int = 60, limit: int = 50):
    db = get_db()
    try:
        jobs = db.get_user_jobs(user_id, min_score=min_score, limit=limit)
        return JSONResponse({"jobs": jobs, "count": len(jobs)})
    finally:
        db.close()


@app.get("/api/profile/{user_id}")
async def api_get_profile(user_id: int):
    db = get_db()
    try:
        profile = db.get_user_profile(user_id)
        return JSONResponse({"profile": profile})
    finally:
        db.close()


@app.post("/api/discount/validate")
async def api_validate_discount(code: str = Form(...), user_id: int = Form(...)):
    db = get_db()
    try:
        result = db.validate_discount(code, user_id)
        return JSONResponse(result)
    finally:
        db.close()


@app.post("/api/discount/apply")
async def api_apply_discount(code: str = Form(...), user_id: int = Form(...)):
    db = get_db()
    try:
        result = db.apply_discount(user_id, code)
        return JSONResponse(result)
    finally:
        db.close()


# ---- Stripe Webhooks & Checkout ----

@app.post("/api/checkout/{user_id}")
async def api_create_checkout(user_id: int, plan: str = Form("pro"),
                               discount_code: str = Form("")):
    """Create a Stripe checkout session."""
    try:
        import stripe
        stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
        if not stripe.api_key:
            raise HTTPException(status_code=500, detail="Stripe not configured")
    except ImportError:
        raise HTTPException(status_code=500, detail="Stripe not installed")

    db = get_db()
    try:
        user = db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        price_map = {
            "pro": os.environ.get("STRIPE_PRO_PRICE_ID", ""),
            "enterprise": os.environ.get("STRIPE_ENTERPRISE_PRICE_ID", ""),
        }
        price_id = price_map.get(plan)
        if not price_id:
            raise HTTPException(status_code=400, detail="Invalid plan")

        base_url = os.environ.get("BASE_URL", "http://localhost:8000")

        session_params = {
            "payment_method_types": ["card"],
            "line_items": [{"price": price_id, "quantity": 1}],
            "mode": "subscription",
            "success_url": f"{base_url}/dashboard/{user_id}?upgraded=true",
            "cancel_url": f"{base_url}/dashboard/{user_id}",
            "client_reference_id": str(user_id),
            "customer_email": user["email"],
        }

        # Apply percentage discount via Stripe coupon
        if discount_code:
            result = db.validate_discount(discount_code, user_id)
            if result["valid"] and result["discount"]["discount_type"] == "percent_off":
                # Create a one-time Stripe coupon
                coupon = stripe.Coupon.create(
                    percent_off=result["discount"]["value"],
                    duration="once",
                )
                session_params["discounts"] = [{"coupon": coupon.id}]

        session = stripe.checkout.Session.create(**session_params)
        return JSONResponse({"checkout_url": session.url})
    finally:
        db.close()


@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    try:
        import stripe
        stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    except ImportError:
        raise HTTPException(status_code=500)

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig, endpoint_secret)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    db = get_db()
    try:
        if event["type"] == "checkout.session.completed":
            session = event["data"]["object"]
            user_id = int(session["client_reference_id"])
            sub_id = session.get("subscription", "")

            # Determine plan from price
            plan = "pro"  # Default
            if sub_id:
                sub = stripe.Subscription.retrieve(sub_id)
                price_id = sub["items"]["data"][0]["price"]["id"]
                if price_id == os.environ.get("STRIPE_ENTERPRISE_PRICE_ID"):
                    plan = "enterprise"

            db.set_plan(user_id, plan, stripe_sub_id=sub_id)
            logger.info(f"User {user_id} upgraded to {plan}")

        elif event["type"] == "customer.subscription.deleted":
            sub = event["data"]["object"]
            # Find user by subscription ID
            rows = db.conn.execute(
                "SELECT id FROM users WHERE stripe_subscription_id = ?",
                (sub["id"],),
            ).fetchone()
            if rows:
                db.set_plan(rows["id"], "free", stripe_sub_id="")
                logger.info(f"User {rows['id']} downgraded to free")

        return JSONResponse({"received": True})
    finally:
        db.close()


# ---- Admin endpoints (protect in production) ----

@app.post("/api/admin/create-discount")
async def admin_create_discount(
    code: str = Form(...),
    discount_type: str = Form(...),
    value: float = Form(...),
    max_uses: int = Form(None),
    expires_days: int = Form(None),
    admin_key: str = Form(...),
):
    if admin_key != os.environ.get("ADMIN_API_KEY", ""):
        raise HTTPException(status_code=403, detail="Unauthorized")

    db = get_db()
    try:
        expires_at = None
        if expires_days:
            expires_at = (datetime.utcnow() + timedelta(days=expires_days)).isoformat()

        disc = db.create_discount(
            code=code,
            discount_type=discount_type,
            value=value,
            max_uses=max_uses,
            expires_at=expires_at,
        )
        return JSONResponse({"success": True, "discount": disc})
    finally:
        db.close()
