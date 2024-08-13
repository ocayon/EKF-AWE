import run_from_csv
from pathlib import Path
from datetime import datetime


if __name__ == "__main__":
    awes_model = run_from_csv.AWESModel(model_name='v3', log_provider=run_from_csv.LogProvider.Kitepower)
    date = datetime.strptime('2019-10-08', '%Y-%m-%d')
    analysis_mode = run_from_csv.AnalyzeAweFromCsvLog.AnalysisMode.Analyze
    default_log_dir = Path('./data/v3/')

    run_from_csv.AnalyzeAweFromCsvLog(awes_model=awes_model,
                                      date=date,
                                      analysis_mode=analysis_mode,
                                      log_directory=default_log_dir)
