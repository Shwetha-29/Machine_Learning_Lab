import re
from datetime import datetime

# ===============================
# Student Class (OOP)
# ===============================
class Student:
    def __init__(self, sid, name):
        self.sid = sid
        self.name = name
        self.login_count = 0
        self.submit_count = 0
        self.is_logged_in = False   # for abnormal behavior
        self.abnormal = False

    def add_activity(self, activity):
        if activity == "LOGIN":
            self.login_count += 1
            if self.is_logged_in:        # LOGIN again without LOGOUT
                self.abnormal = True
            self.is_logged_in = True

        elif activity == "LOGOUT":
            self.is_logged_in = False

        elif activity == "SUBMIT_ASSIGNMENT":
            self.submit_count += 1

    def display(self):
        print("Student ID:", self.sid)
        print("Name:", self.name)
        print("Total Logins:", self.login_count)
        print("Total Submissions:", self.submit_count)
        if self.abnormal:
            print("⚠ Abnormal Behavior: Multiple logins without logout")
        print()


# ===============================
# Step 1: Real-time logging
# ===============================
LOG_FILE = "student_log.txt"

print("\n=== REAL-TIME STUDENT ACTIVITY LOGGER ===")
print("Allowed activities: LOGIN, LOGOUT, SUBMIT_ASSIGNMENT")
print("Type 'stop' as Student ID to finish\n")

with open(LOG_FILE, "a") as file:
    while True:
        sid = input("Student ID: ").strip()

        if sid.lower() == "stop":
            break

        name = input("Student Name: ").strip()
        activity = input("Activity: ").upper().strip()

        try:
            # Validation
            if not re.fullmatch(r"S\d+", sid):
                raise ValueError("Invalid Student ID")

            if activity not in ["LOGIN", "LOGOUT", "SUBMIT_ASSIGNMENT"]:
                raise ValueError("Invalid Activity")

            # Automatic date & time
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            file.write(f"{sid} | {name} | {activity} | {date} | {time}\n")
            print("✅ Activity logged successfully!\n")

        except Exception as e:
            print("❌ Error:", e, "\n")

print("\nLogging completed.\n")


# ===============================
# Step 2: Generator to read logs
# ===============================
def read_logs(filename):
    with open(filename, "r") as file:
        for line in file:
            try:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) != 5:
                    raise ValueError("Invalid log format")

                sid, name, activity, date, time = parts

                if not re.fullmatch(r"S\d+", sid):
                    raise ValueError("Invalid Student ID")

                if activity not in ["LOGIN", "LOGOUT", "SUBMIT_ASSIGNMENT"]:
                    raise ValueError("Invalid Activity")

                yield sid, name, activity, date

            except Exception:
                print("Skipping invalid log entry:", line.strip())


# ===============================
# Step 3: Process logs
# ===============================
students = {}
daily_stats = {}

for sid, name, activity, date in read_logs(LOG_FILE):

    if sid not in students:
        students[sid] = Student(sid, name)

    students[sid].add_activity(activity)

    # Daily activity statistics
    if date not in daily_stats:
        daily_stats[date] = 0
    daily_stats[date] += 1


# ===============================
# Step 4: Generate report
# ===============================
if not students:
    print("⚠ No valid student records found. Report not generated.")
else:
    with open("activity_report.txt", "w") as report:

        print("=== STUDENT ACTIVITY REPORT ===\n")
        report.write("=== STUDENT ACTIVITY REPORT ===\n\n")

        for student in students.values():
            student.display()

            report.write(f"Student ID: {student.sid}\n")
            report.write(f"Name: {student.name}\n")
            report.write(f"Total Logins: {student.login_count}\n")
            report.write(f"Total Submissions: {student.submit_count}\n")

            if student.abnormal:
                report.write("Abnormal Behavior: Multiple logins without logout\n")

            report.write("\n")

        # Daily statistics
        print("=== DAILY ACTIVITY STATISTICS ===")
        report.write("=== DAILY ACTIVITY STATISTICS ===\n")

        for date, count in daily_stats.items():
            print(date, ":", count, "activities")
            report.write(f"{date}: {count} activities\n")

    print("\n✅ Report generated successfully!")