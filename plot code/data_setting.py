import sys
sys.path.append('src')
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

__all__ = ["get_weekday_df"]

from pathlib import Path


def get_weekday_df(file_path: str | None = None, cache: bool = True) -> pd.DataFrame:
	"""Load small office hourly data from `file_path` and return a processed weekday_df.

	Behavior mirrors the original script:
	- strips Date/Time whitespace
	- computes Day (0=Mon .. 6=Sun) from row index assuming 24 rows/day with 1 Jan = Sunday
	- zeroes weekend loads
	- zeroes cooling in heating months and heating in cooling months

	Args:
		file_path: path to the directory containing 'small_office_hour.csv' (default 'data/')

	Returns:
		Processed pandas.DataFrame
	"""
	# Determine data directory: caller can pass file_path or we'll auto-resolve
	if file_path is None:
		# Resolve relative to project root: assume this file is in '<repo>/.../plot code/'
		repo_root = Path(__file__).resolve().parents[2]
		data_dir = repo_root / 'data'
	else:
		data_dir = Path(file_path)

	data_dir = data_dir if data_dir.is_dir() else data_dir.parent

	# Use cached processed file if available
	processed_dir = data_dir / 'processed'
	processed_file = processed_dir / 'small_office_ready.csv'
	raw_file = data_dir / 'small_office_hour.csv'

	if cache and processed_file.exists():
		small_office_df = pd.read_csv(processed_file)
	else:
		small_office_df = pd.read_csv(raw_file)

	# 1. Date/Time 열 공백 정리
	weekday_df = small_office_df.copy()
	weekday_df['Date/Time_clean'] = weekday_df['Date/Time'].str.strip().str.replace(r'\s+', ' ', regex=True)

	# 2. 시간 인덱스 기반으로 '요일' 계산 (1월 1일 = 일요일)
	weekday_df['Relative_Day_Index'] = weekday_df.index // 24
	weekday_df['Day'] = (6 + weekday_df['Relative_Day_Index']) % 7

	# 3. 평일 데이터만 필터링 (Weekday 0~4 → 월~금)
	weekday_df.loc[weekday_df['Day'].isin([5, 6]), 'DistrictCooling:Facility [J](TimeStep)'] = 0
	weekday_df.loc[weekday_df['Day'].isin([5, 6]), 'DistrictHeatingWater:Facility [J](TimeStep) '] = 0

	# 4. 10-3월은 난방만, 4-9월은 냉방만 발생하도록 설정
	# 월 정보 추출
	weekday_df['Month'] = weekday_df['Date/Time_clean'].str.slice(0, 2).astype(int)
	# 난방월에는 냉방 부하를 0으로 설정
	weekday_df.loc[weekday_df['Month'].isin([1, 2, 3, 10, 11, 12]), 'DistrictCooling:Facility [J](TimeStep)'] = 0
	# 냉방월에는 난방 부하를 0으로 설정
	weekday_df.loc[weekday_df['Month'].isin([4, 5, 6, 7, 8, 9]), 'DistrictHeatingWater:Facility [J](TimeStep) '] = 0

	# 6. 인덱스 초기화
	weekday_df.reset_index(drop=True, inplace=True)

	# Save processed file for faster subsequent loads
	try:
		if cache:
			processed_dir.mkdir(exist_ok=True)
			weekday_df.to_csv(processed_file, index=False)
	except Exception:
		# don't fail if saving cache fails
		pass

	return weekday_df

if __name__ == '__main__':
	# quick smoke test when run directly
	df = get_weekday_df('data/')
	print('Loaded weekday_df with shape:', df.shape)