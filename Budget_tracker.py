import datetime

transactions = []

def add_income(amount, description="Income"):
    transactions.append({
        "type": "income", 
        "amount": amount, 
        "description": description, 
        "date": datetime.datetime.now()
    })

def add_expense(amount, description="Expense"):
    transactions.append({
        "type": "expense", 
        "amount": amount, 
        "description": description, 
        "date": datetime.datetime.now()
    })

def show_summary():
    total_income = sum(t["amount"] for t in transactions if t["type"] == "income")
    total_expense = sum(t["amount"] for t in transactions if t["type"] == "expense")
    balance = total_income - total_expense
    print(f"Total Income: {total_income}")
    print(f"Total Expense: {total_expense}")
    print(f"Balance: {balance}")

def show_transactions():
    for t in transactions:
        print(f"{t['date'].strftime('%Y-%m-%d %H:%M:%S')} - {t['description']}: {t['amount']} ({t['type']})")

def main():
    while True:
        print("\nBudget Tracker")
        print("1. Add Income")
        print("2. Add Expense")
        print("3. Show Summary")
        print("4. Show Transactions")
        print("5. Exit")
        
        choice = input("Choose an option: ")
        
        if choice == "1":
            try:
                amount = float(input("Enter income amount: "))
                description = input("Enter income description: ")
                add_income(amount, description)
            except ValueError:
                print("Invalid amount. Please enter a number.")
        elif choice == "2":
            try:
                amount = float(input("Enter expense amount: "))
                description = input("Enter expense description: ")
                add_expense(amount, description)
            except ValueError:
                print("Invalid amount. Please enter a number.")
        elif choice == "3":
            show_summary()
        elif choice == "4":
            show_transactions()
        elif choice == "5":
            break
        else:
            print("Invalid choice. Please try again.")


main()
