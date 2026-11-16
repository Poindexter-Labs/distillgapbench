"""Print `name repo revision` for students, in the fixed v1 order (used by serve_students.sh)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distillgap.students import STUDENT_ORDER, STUDENTS  # noqa: E402

wanted = sys.argv[1:] or STUDENT_ORDER
for name in STUDENT_ORDER:
    if name in wanted:
        print(name, STUDENTS[name]["repo"], STUDENTS[name]["revision"])
unknown = sorted(set(wanted) - set(STUDENT_ORDER))
if unknown:
    sys.exit("unknown students: " + " ".join(unknown))
