# Mine Location Data Directory

This directory contains CSV files with mine location data. Each file should follow the format:

```csv
latitude,longitude,status
12.3456,78.9012,active
23.4567,89.0123,defused
```

File naming convention: `mine_locations_YYYYMMDD_HHMMSS.csv`

## Columns:
- latitude: Decimal degrees (float)
- longitude: Decimal degrees (float)
- status: String ('active' or 'defused')