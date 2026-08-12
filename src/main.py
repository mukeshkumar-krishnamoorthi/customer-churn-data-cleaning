import sys

from ingestion.csv_loader import CSVLoader
from profiling.profiler import DataProfiler
from utils.config import load_json_config
from utils.logger import Logger
from utils.writer import FileWriter
from validation.data_quality import DataQuality
from cleaning.data_cleaner import DataCleaner

RAW_DATA_PATH = "data/raw/IT_customer_churn.csv"
QUALITY_RULES_PATH = "config/quality_rules.json"
CLEANING_RULES_PATH = "config/cleaning_rules.json"
QUALITY_GATE_MIN_SCORE = 70

df = CSVLoader(RAW_DATA_PATH).load()

# --- Profiling ---
profiler = DataProfiler(df)
profiler.print_report()

FileWriter.csv(profiler.column_summary(), "outputs/reports/column_summary.csv")
FileWriter.json(profiler.to_report(), "outputs/reports/profile_report.json")

# --- Data quality ---
quality_rules = load_json_config(QUALITY_RULES_PATH)
quality_report = DataQuality(df, quality_rules).run()

Logger.info(
    "Data Quality Report",
    Score=quality_report.score,
    Errors=quality_report.error_count,
    Warnings=quality_report.warning_count,
)

for issue in quality_report.issues:
    log = Logger.error if issue.severity == "error" else Logger.warning
    log(f"[{issue.check}] {issue.message}")

FileWriter.json(quality_report.to_dict(),
                "outputs/reports/quality_report.json")
quality_report.to_dataframe().to_csv(
    "outputs/reports/quality_issues.csv", index=False)

if quality_report.score < QUALITY_GATE_MIN_SCORE:
    Logger.error(
        f"Quality gate failed: score {quality_report.score} < {QUALITY_GATE_MIN_SCORE}")
    sys.exit(1)

Logger.success(f"Quality gate passed: score {quality_report.score}")

cleaning_rules = load_json_config(CLEANING_RULES_PATH)
columns = DataCleaner(df, cleaning_rules)
clean_df = columns.clean()
print(clean_df)

