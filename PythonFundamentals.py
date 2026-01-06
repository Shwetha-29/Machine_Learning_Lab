# Tuple (student IDs)
student_ids = ('S101', 'S102', 'S103', 'S104')

# Dictionary (student records)
students = {
    'S101': {'name': 'Asha', 'assignment': 78, 'test': 80, 'attendance': 92, 'hours': 8},
    'S102': {'name': 'Ravi', 'assignment': 65, 'test': 68, 'attendance': 85, 'hours': 5},
    'S103': {'name': 'Meena', 'assignment': 88, 'test': 90, 'attendance': 96, 'hours': 10},
    'S104': {'name': 'Kiran', 'assignment': 55, 'test': 58, 'attendance': 78, 'hours': 4}
}


def calculate_average(assignment, test):
    return (assignment + test) / 2

def risk_level(avg, attendance, hours):
    if avg >= 75 and attendance >= 90 and hours >= 7:
        return "Low Risk"
    elif avg >= 60 and attendance >= 80:
        return "Medium Risk"
    else:
        return "High Risk"
for sid in student_ids:
    s = students[sid]
    avg = calculate_average(s['assignment'], s['test'])
    risk = risk_level(avg, s['attendance'], s['hours'])

    print("\nStudent ID:", sid)
    print("Name:", s['name'])
    print("Average Score:", avg)
    print("Attendance:", s['attendance'])
    print("Study Hours:", s['hours'])
    print("Academic Risk:", risk)