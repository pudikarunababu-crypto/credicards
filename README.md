# 💳 CC Info · ML Dataset Explorer

A clean, interactive Streamlit dashboard for exploring the `cc_info` credit card dataset used in ML model training.

## Features

- **Overview** — credit limit distribution, box plot, and avg limit by state
- **Geographic** — US choropleth maps (record count + avg limit), top cities
- **Data Table** — searchable, filterable raw data with masked card numbers + CSV download
- **ML Insights** — feature stats, cardinality, percentile curve, and training notes

## Project Structure

```
├── app.py                  # Main Streamlit application
├── cc_info.csv             # Dataset (add this file — not committed to git)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Theme & server config
└── README.md
```

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure cc_info.csv is in the root folder

# 4. Launch
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo (including `cc_info.csv`) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo / branch / `app.py`
4. Click **Deploy** — done!

> ⚠️ `cc_info.csv` must be committed to the repo so Streamlit Cloud can read it. If the file contains real card data, sanitize or anonymize it first.

## Dataset Schema

| Column | Type | Description |
|---|---|---|
| `credit_card` | int64 | 16-digit card number |
| `city` | string | Cardholder city |
| `state` | string | US state abbreviation |
| `zipcode` | int64 | ZIP code |
| `credit_card_limit` | int64 | Credit limit in USD |

984 records · 35 unique states · 124 unique cities · no nulls
