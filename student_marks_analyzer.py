import numpy as np
marks=np.array([
    [80,75,98],
    [65,90,72],
    [88,92,85],
    [70,80,75]
    ])
total_per_student = np.sum(marks, axis=1)
print("Total marks per student:", total_per_student)

avarage_per_subject = np.mean(marks, axis=0)
print("Average marks per subject:", avarage_per_subject)

topper_student = np.argmax(total_per_student)
print(f"Topper: Student {topper_student +1}")

weakest_subject = np.argmin(avarage_per_subject)
print(f"Weakest subject: Subject {weakest_subject +1}")