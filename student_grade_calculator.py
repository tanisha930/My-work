class Student:
    def __init__(self,name):
        self.name=name
        self.subjects={}
    def add_subject(self,subject_name,mark):
        self.subjects[subject_name] = mark   
    def calculate_gpa(self):
        total_marks=sum(self.subjects.values())  
        subject_count= len(self.subjects) 
        average=total_marks/subject_count
        
        if average >=80:
            grade='A+'
            gpa=4.0
        elif average >=70:
            grade='A'
            gpa=3.5
        elif average >=60:
            grade='A-'
            gpa=3.0
        elif average>= 50:
            grade='B'
            gpa=2.5
        else:
            grade='f'
            gpa=0.0
        return gpa,grade   
    
    #input user
    name=input("Enter your name:")
    subject_count=int(input("how many subjects?"))
    subjects={} 
    for i in range(subject_count):
        subject=input(f"Enter name of subjects(i+1):")
        mark=float(input(f"Enter marks for{subject}:"))
        subjects[subject]=mark
    #GPA counting    
    gpa,grade= calculate_gpa(list(subjects.values())) 
    
    #output
    output=f"Name:{name}" 
    for subject, mark in subjects.items():
        output += f"{subject}:{mark}"
    output +=f"GPA:{gpa} Grade:{grade}"
    #print
    print(output)    
    with open(f"{name}_result.txt","w")as f:
        f.write(output)     
                
    