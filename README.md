# Job Search Agent

Automated job discovery agent that scrapes 20+ sources, scores jobs with Claude AI against your profile, verifies application links, and delivers curated results via Gmail and Telegram.

## Quick Start

### 1. Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts to name your bot
3. Copy the **bot token** (looks like `123456789:ABCdefGHI...`)
4. Send any message to your new bot (just say "hi")
5. Get your **chat_id** by visiting:
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
   Look for `"chat":{"id":123456789}` — that number is your chat_id

### 2. Create a Resend Account

1. Go to [resend.com](https://resend.com) and sign up (free)
2. Get your **API key** from the dashboard
3. For testing, you can use the default `onboarding@resend.dev` sender
4. For production, add and verify your own sending domain

### 3. Set Up the GitHub Repository

1. Create a **public** repository on GitHub (unlimited Actions minutes)
2. Go to **Settings → Secrets and variables → Actions**
3. Add these **repository secrets**:

   | Secret Name | Value |
   |---|---|
   | `ANTHROPIC_API_KEY` | Your Claude API key from console.anthropic.com |
   | `RESEND_API_KEY` | Your Resend API key |
   | `TELEGRAM_BOT_TOKEN` | Bot token from BotFather |
   | `TELEGRAM_CHAT_ID` | Your chat ID from step 1.5 |

4. Push this code to the repository
5. The agent starts running automatically on schedule

### 4. Test Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY="your-key"
export RESEND_API_KEY="your-key"
export TELEGRAM_BOT_TOKEN="your-token"
export TELEGRAM_CHAT_ID="your-chat-id"

# Dry run (scrape + analyze, no notifications)
python -m src.main --dry-run --hours 72

# Full run with notifications
python -m src.main --email --telegram --hours 24
```

## How It Works

**Every 4 hours** the agent:
1. Scrapes LinkedIn, Indeed, Glassdoor, ZipRecruiter, Google Jobs (via JobSpy)
2. Scrapes AI-specific boards: YC Work at a Startup, aijobs.com, Built In, Remotive
3. Scrapes company career pages: Anthropic, OpenAI, DeepMind, Cohere, Mistral, HuggingFace, Nvidia
4. Deduplicates against seen jobs (SQLite)
5. Validates every URL (HEAD check → content verification)
6. Sends new jobs to Claude Haiku for scoring (0-100) against your profile
7. Stores results

**Twice daily** (6AM + 2PM EAT):
- **6AM**: Gmail digest + Telegram alerts (APPLY + MAYBE jobs)
- **2PM**: Telegram alerts only (new jobs since morning)

## Schedule

| Run | Time (EAT) | UTC | Notifications |
|---|---|---|---|
| 1 | 2:00 AM | 23:00 | Silent scrape |
| 2 | 6:00 AM | 03:00 | Email + Telegram |
| 3 | 10:00 AM | 07:00 | Silent scrape |
| 4 | 2:00 PM | 11:00 | Telegram only |
| 5 | 6:00 PM | 15:00 | Silent scrape |
| 6 | 10:00 PM | 19:00 | Silent scrape |

## Customization

Edit `config/profile.yaml` to update your skills, target roles, and preferences.

Edit `config/search_queries.yaml` to change search terms.

## Monthly Cost

| Service | Cost |
|---|---|
| GitHub Actions | Free (public repo) |
| Claude API (Haiku) | ~$2-5 |
| Resend | Free (3K emails/mo) |
| Telegram | Free |
| **Total** | **~$2-5/month** |
