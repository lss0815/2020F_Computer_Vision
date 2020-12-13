"""import numpy as np

a = np.array([np.float(0), np.float('inf'), np.float('-inf'), np.float('NaN')])

a = np.nan_to_num(a, nan=np.float64(0), posinf=np.float64('inf'), neginf=np.float64('-inf'))

print(a)

print(np.arctan(a))"""

import datetime

time1 = datetime.date(2020,2,29)
time2 = datetime.date(time1.year + 1, time1.month, time1.day-1)

print(time1)