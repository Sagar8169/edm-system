"""
student_names.py — Helper to generate deterministic, realistic student metadata (Names, CGPA, Attendance)
and course module display labels for the EDM system.
"""

import numpy as np
import pandas as pd

FIRST_NAMES = [
    "Aarav", "Ananya", "Rohan", "Priya", "Aditya", "Sneha", "Rahul", "Pooja", "Vikram", "Neha",
    "Karan", "Kavya", "Siddharth", "Riya", "Arjun", "Anushka", "Dev", "Isha", "Manish", "Divya",
    "Amit", "Tanvi", "Suresh", "Meera", "Yash", "Simran", "Rajesh", "Kiran", "Nikhil", "Shweta",
    "Abhishek", "Deepika", "Varun", "Shruti", "Gaurav", "Nisha", "Rishabh", "Preeti", "Kunal", "Swati",
    "Alex", "Emma", "Liam", "Sophia", "Noah", "Olivia", "Ethan", "Ava", "Lucas", "Mia"
]

LAST_NAMES = [
    "Sharma", "Verma", "Mishra", "Gupta", "Singh", "Patel", "Kumar", "Shah", "Joshi", "Mehta",
    "Chawla", "Deshmukh", "Nair", "Iyer", "Reddy", "Rao", "Malhotra", "Kapoor", "Bhat", "Agarwal",
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson", "Taylor", "Anderson"
]

MODULE_NAME_MAP = {
    "AAA": "1 - BDA (Big Data Analytics)",
    "BBB": "2 - ML (Machine Learning)",
    "CCC": "3 - WAIR (Web AI & Robotics)",
    "DDD": "1 - BDA (Big Data Analytics)",
    "EEE": "2 - ML (Machine Learning)",
    "FFF": "3 - WAIR (Web AI & Robotics)",
    "GGG": "1 - BDA (Big Data Analytics)",
    "0": "1 - BDA (Big Data Analytics)",
    "1": "2 - ML (Machine Learning)",
    "2": "3 - WAIR (Web AI & Robotics)",
    "3": "1 - BDA (Big Data Analytics)",
    "4": "2 - ML (Machine Learning)",
    "5": "3 - WAIR (Web AI & Robotics)",
    "6": "1 - BDA (Big Data Analytics)",
}


def get_course_display_name(mod):
    mod_str = str(mod).strip()
    return MODULE_NAME_MAP.get(mod_str, f"{mod_str} - Elective")


def generate_student_info(student_id: int, avg_score: float = 75.0, days_active: float = 40.0):
    """Generate realistic Name, CGPA, Attendance % deterministically based on student_id."""
    np.random.seed(int(student_id) % 1000000)
    
    first = FIRST_NAMES[int(student_id) % len(FIRST_NAMES)]
    last = LAST_NAMES[(int(student_id) // 7) % len(LAST_NAMES)]
    name = f"{first} {last}"
    
    # Calculate realistic CGPA (2.0 to 10.0 scale) correlated with avg_score
    score_factor = max(30.0, min(100.0, float(avg_score if not pd.isna(avg_score) else 70.0)))
    base_cgpa = (score_factor / 100.0) * 8.5 + 1.2
    cgpa_noise = np.random.uniform(-0.4, 0.4)
    cgpa = round(float(np.clip(base_cgpa + cgpa_noise, 4.0, 9.9)), 2)
    
    # Calculate Attendance % correlated with days_active
    days_factor = float(days_active if not pd.isna(days_active) else 45.0)
    base_att = min(98.0, max(45.0, (days_factor / 80.0) * 85.0 + 10.0))
    att_noise = np.random.uniform(-5.0, 5.0)
    attendance = round(float(np.clip(base_att + att_noise, 35.0, 99.0)), 1)
    
    return {
        "student_name": name,
        "cgpa": cgpa,
        "attendance_pct": attendance
    }
