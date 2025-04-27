import pandas as pd
from ydata_profiling import ProfileReport
import os

class DataProfiler:
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.input_file_path = f'data/{file_name}'
        self.output_dir = 'output'
        os.makedirs(self.output_dir, exist_ok=True)  # Create output folder if it doesn't exist
        self.df = self._load_data()
        self.report = self._create_report()
    
    def _load_data(self):
        return pd.read_csv(self.input_file_path, encoding='utf-8')

    def _create_report(self):
        return ProfileReport(self.df, title='Profiling Report')
    
    def save_report(self):
        output_path = f'{self.output_dir}/report - {self.file_name}'
        self.report.to_file(output_path)

    def show_report(self):
        return self.report

# data_profiler_atlas = DataProfiler('Atlas Cechu Student Access.csv')
# data_profiler_atlas.save_report()

# data_profiler_payments = DataProfiler('Payments Student Access.csv')
# data_profiler_payments.save_report()

data_profiler_credits = DataProfiler('User Credits Student Access.csv')
data_profiler_credits.save_report()