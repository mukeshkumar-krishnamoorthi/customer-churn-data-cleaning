from ingestion.csv_loader import CSVLoader
from profiling.profiler import DataProfiler
from validation.data_quality import DataQuality

loader = CSVLoader("data/raw/IT_customer_churn.csv")

df = loader.load()

profiler = DataProfiler(df)

report = profiler.column_summary()

# Print to console
print(report)

# Save to file
profiler.save_as_csv(report)
