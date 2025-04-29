contacts={}
def add_contacts():
    name=input("enter contact name:")
    phone=input("enter phone number:")
    email=input("enter email address:")
    contacts[name]={"Phone":phone, "Email":email}
    print(f"{name} added successfully.\n")
    
def view_contacts():
    if not contacts:
        print("No contacts found.\n") 
    else:
        print("contacts list ")   
        for name,info in contacts.items():
            print(f"Name:{name}") 
            print(f"Phone:{info['Phone']}")  
            print(f"Email{info['Email']}\n")
 
def search_contacts():
    name=input("Enter name to search: ")  
    if name in contacts:
        print(f"name:{name}") 
        print(f"Phone:{contacts[name]['Phone']}") 
        print(f"Email:{contacts[name]['Email']}\n") 
    else:
        print("contact not found\n") 
        
def delete_contacts():
    name=input("Enter name to delete:") 
    if name in contacts:
        del contacts[name]  
        print(f"{name} deleted successfully.\n") 
    else:
        print("contact not found.\n")  
        
        
while True:
    print("contact book manu:") 
    print("1.Add contact:") 
    print("2.View contacts") 
    print("3.Search contact") 
    print("4.delete contact") 
    print("5.Exit")  
    choice= input("Enter your choice(1-5)") 
    
    if choice=='1':
        add_contacts() 
    elif choice=='2':
        view_contacts()
    elif choice=='3': 
        search_contacts()
    elif choice=='4':
        delete_contacts() 
    elif choice=='5':
        print("Exiting contact book.")
        break
    else:
        print("invalid choice.\n")
                        
            