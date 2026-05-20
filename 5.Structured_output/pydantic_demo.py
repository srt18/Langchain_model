from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str 
    age: Optional[int] = None
    email: EmailStr
    cgpa: Optional[float] = Field(None, ge=0.0, le=4.0, description="Grade Point Average")  # GPA between 0.0 and 4.0

new_student = Student(name="Alice", email="alice@example.com", age=30, cgpa=3.5)

print(new_student) # <class '__main__.Student'>

# dict representation
new_student_dict = dict(new_student)
print(new_student_dict) # {'name': 'Alice', 'age': 30, 'email': ''}

#json representation
new_student_json = new_student.model_dump_json()
print(new_student_json) # {"name": "Alice", "age": 30}