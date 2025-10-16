import sys
sys.path.append('src')
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

__all__ = ["get_weekday_df"]

from pathlib import Path


def _find_column(columns, key: str):
	"""Find a column name in `columns` that matches `key` approximately.

	Matching strategy (in order):
	- exact strip match (c.strip() == key)
	- contains key substring
	- startswith key
	Returns the actual column name (may include trailing spaces) or None.
	"""
	for c in columns:
		if c.strip() == key:
			return c
	for c in columns:
		if key in c:
			return c
	for c in columns:
		if c.startswith(key):
			return c
	return None


def find_repo_root(start: Path | None = None) -> Path:
	"""Walk ancestors from start (default this file) and return the first directory
	that looks like the repository root (contains 'data', 'pyproject.toml' or '.git').
	Falls back to the top-most ancestor if nothing is found.
	"""
	if start is None:
		start = Path(__file__).resolve()
	for parent in [start] + list(start.parents):
		if (parent / 'data').exists() or (parent / 'pyproject.toml').exists() or (parent / '.git').exists():
			return parent
	# fallback to highest ancestor
	return start.parents[-1]


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
	# Determine data directory: caller can pass file_path (dir or file) or we'll auto-resolve
	if file_path is None:
		repo_root = find_repo_root()
		data_dir = repo_root / 'data'
		raw_file = data_dir / 'small_office_hour.csv'
	else:
		p = Path(file_path)
		# If the provided path doesn't exist, try resolving it relative to the repo root
		repo_root = find_repo_root()
		candidates = [
			p,
			repo_root / file_path,
			repo_root / 'data' / p.name,
			repo_root / 'data' / 'small_office_hour.csv',
			Path.cwd() / file_path,
		]
		found = None
		for c in candidates:
			if c.exists():
				found = c
				break
		# Fallback: if nothing exists, keep original p so error reporting shows attempted locations
		if found is None:
			# We'll set data_dir to the original p (likely a directory) and let the existence check fail later
			data_dir = p
			raw_file = data_dir / 'small_office_hour.csv'
		else:
			if found.is_file():
				raw_file = found
				data_dir = found.parent
			else:
				data_dir = found
				raw_file = data_dir / 'small_office_hour.csv'

	processed_dir = data_dir / 'processed'
	processed_file = processed_dir / 'small_office_ready.csv'

	# 1. processed 파일이 있으면 바로 반환 (단, 컬럼은 원본 형태를 유지하지만
	#    필요한 컬럼은 확인/강제변환함)
	if cache and processed_file.exists():
		small_office_df = pd.read_csv(processed_file)
		# detect important columns but preserve their exact names (possibly with trailing space)
		cols = list(small_office_df.columns)
		date_col = _find_column(cols, 'Date/Time')
		heating_col = _find_column(cols, 'DistrictHeatingWater:Facility')
		cooling_col = _find_column(cols, 'DistrictCooling:Facility')

		# coerce numeric for load columns if present
		if heating_col is not None:
			small_office_df[heating_col] = pd.to_numeric(small_office_df[heating_col], errors='coerce')
		if cooling_col is not None:
			small_office_df[cooling_col] = pd.to_numeric(small_office_df[cooling_col], errors='coerce')
		return small_office_df

	# 2. 없으면 data 폴더의 small_office_hour 파일을 불러와서 전처리
	if raw_file.exists():
		small_office_df = pd.read_csv(raw_file)

	else:
		tried = [str(processed_file), str(raw_file)]
		raise FileNotFoundError(
			"Could not find input data. Tried the following paths:\n  " + "\n  ".join(tried) +
			"\n\nProvide the directory containing 'small_office_hour.csv' or pass the full path to the CSV file to get_weekday_df(file_path)."
	)

	# Normalize and detect columns, but preserve the original column names so that
	# heating column can keep its trailing space if present in the CSV header.
	weekday_df = small_office_df.copy()
	cols = list(weekday_df.columns)
	date_col = _find_column(cols, 'Date/Time')
	heating_col = _find_column(cols, 'DistrictHeatingWater:Facility')
	cooling_col = _find_column(cols, 'DistrictCooling:Facility')

	if date_col is None:
		raise KeyError("Expected 'Date/Time' column in input CSV; available columns: " + ",".join(list(weekday_df.columns)))
	# create a cleaned Date/Time string column
	weekday_df['Date/Time_clean'] = weekday_df[date_col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

	# 2. 시간 인덱스 기반으로 '요일' 계산 (1월 1일 = 일요일)
	weekday_df['Relative_Day_Index'] = weekday_df.index // 24
	weekday_df['Day'] = (6 + weekday_df['Relative_Day_Index']) % 7

	# 3. 평일 데이터만 필터링 (Weekday 0~4 → 월~금)
	if cooling_col is not None:
		weekday_df.loc[weekday_df['Day'].isin([5, 6]), cooling_col] = 0
	if heating_col is not None:
		weekday_df.loc[weekday_df['Day'].isin([5, 6]), heating_col] = 0

	# 4. 10-3월은 난방만, 4-9월은 냉방만 발생하도록 설정
	# 월 정보 추출
	weekday_df['Month'] = weekday_df['Date/Time_clean'].str.slice(0, 2).astype(int)
	# 난방월에는 냉방 부하를 0으로 설정
	if cooling_col is not None:
		weekday_df.loc[weekday_df['Month'].isin([1, 2, 3, 10, 11, 12]), cooling_col] = 0
	# 냉방월에는 난방 부하를 0으로 설정
	if heating_col is not None:
		weekday_df.loc[weekday_df['Month'].isin([4, 5, 6, 7, 8, 9]), heating_col] = 0

	# 6. 인덱스 초기화
	weekday_df.reset_index(drop=True, inplace=True)

	# coerce load columns to numeric and fill NaN with 0 (safe for downstream numeric ops)
	if heating_col is not None:
		weekday_df[heating_col] = pd.to_numeric(weekday_df[heating_col], errors='coerce').fillna(0.0)
	if cooling_col is not None:
		weekday_df[cooling_col] = pd.to_numeric(weekday_df[cooling_col], errors='coerce').fillna(0.0)

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
	