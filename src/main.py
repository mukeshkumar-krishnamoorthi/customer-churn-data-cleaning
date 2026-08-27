import sys

from src.ingestion.csv_loader import CSVLoader
from src.profiling.profiler import DataProfiler
from src.utils.config import load_json_config
from src.utils.logger import Logger
from src.utils.writer import FileWriter
from src.validation.data_quality import DataQuality
from src.cleaning.data_cleaner import DataCleaner

RAW_DATA_PATH = "data/raw/IT_customer_churn.csv"
CLEANED_DATA_PATH = "data/processed/IT_customer_churn.csv"
QUALITY_RULES_PATH = "config/quality_rules.json"
CLEANING_RULES_PATH = "config/cleaning_rules.json"
VALIDATOR_RULES_PATH = "config/validator_rules.json"
QUALITY_GATE_MIN_SCORE = 70

df = CSVLoader(RAW_DATA_PATH).load()

# --- Profiling ---
profiler = DataProfiler(df)
profiler.print_report()

FileWriter.csv(profiler.column_summary(), "outputs/reports/column_summary.csv")
FileWriter.json(profiler.to_report(), "outputs/reports/profile_report.json")

# --- Before Cleaning Data quality ---
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
                "outputs/reports/before_quality_report.json")
quality_report.to_dataframe().to_csv(
    "outputs/reports/before_quality_issues.csv", index=False)

if quality_report.score < QUALITY_GATE_MIN_SCORE:
    Logger.error(
        f"Quality gate failed: score {quality_report.score} < {QUALITY_GATE_MIN_SCORE}")
    sys.exit(1)

Logger.success(f"Quality gate passed: score {quality_report.score}")

cleaning_rules = load_json_config(CLEANING_RULES_PATH)
clean_df = DataCleaner(df, cleaning_rules).clean()
Logger.success(
    "After cleaned Save the data"
)

FileWriter.csv(clean_df, CLEANED_DATA_PATH)

# --- After cleaning Data quality ---
quality_rules = load_json_config(VALIDATOR_RULES_PATH)
quality_report = DataQuality(clean_df, quality_rules).run()

Logger.info(
    "After Cleaning Data Quality Report",
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
