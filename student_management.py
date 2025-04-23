class Student:
    def _init_(self, name, roll, grade):
        self.name = name
        self.roll = roll
        self.grade = grade

    def to_string(self):
        return f"{self.name},{self.roll},{self.grade}\n"

    @staticmethod
    def from_string(data_line):
        name, roll, grade = data_line.strip().split(",")
        return Student(name, roll, grade)


def save_student(student, filename="student_records.txt"):
    with open(filename, "a") as file:
        file.write(student.to_string())


def load_students(filename="student_records.txt"):
    students = []
    try:
        with open(filename, "r") as file:
            for line in file:
                students.append(Student.from_string(line))
    except FileNotFoundError:
        pass
    return students


def main():
    while True:
        print("\n1. Add Student")
        print("2. View All Students")
        print("3. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            name = input("Enter name: ")
            roll = input("Enter roll: ")
            grade = input("Enter grade: ")
            student = Student(name, roll, grade)
            save_student(student)
            print("Student saved!")
        elif choice == "2":
            students = load_students()
            if not students:
                print("No students found.")
            else:
                for s in students:
                    print(f"Name: {s.name}, Roll: {s.roll}, Grade: {s.grade}")
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "_main_":
    main()