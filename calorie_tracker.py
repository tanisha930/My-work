import matplotlib.pyplot as plt
food_data= { "rice":{"calorie":325, "carbs":70, "fiber":1, "protein":7, "fat":3.4},
            "chicken":{"calorie":239, "carbs":0, "fiber":0, "protein":27, "fat":14},
            "cucumber":{"calorie":16, "carbs":4, "fiber":1, "protein":1, "fat":0},
            "pizza":{"calorie":285, "carbs":36, "fiber":2, "protein":12, "fat":10},
            "burger":{"calorie":354, "carbs":32, "fiber":2, "protein":17, "fat":19},
            "coke":{"calorie":140, "carbs":39, "fiber":0, "protein":0, "fat":0},
            "pineapple":{"calorie":50, "carbs":13, "fiber":1, "protein":0, "fat":0},
            "orange":{"calorie":47, "carbs":12, "fiber":2, "protein":1, "fat":0},
            "biscuit":{"calorie":502, "carbs":63, "fiber":2, "protein":6, "fat":25},
            "chocolate":{"calorie":546, "carbs":61, "fiber":3, "protein":4, "fat":31},
            "yogurt":{"calorie":59, "carbs":4, "fiber":0, "protein":10, "fat":0.4},
            "cheese":{"calorie":402, "carbs":1, "fiber":0, "protein":25, "fat":33},
            "biryani":{"calorie":300, "carbs":45, "fiber":2, "protein":10, "fat":10},}
def bmr_calculator():
    print("Welcome to the Calorie Tracker!")
    gender = input("enter your gender(male/female):").lower()
    name = input("Enter your name: ")
    age = int(input("Enter your age: "))
    weight = float(input("Enter your weight (in kg): "))
    height = float(input("Enter your height (in cm): "))
    if gender == "female":
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.33 * age)
    else:
        bmr = 88.362 + (13.7 * weight) + (4.799 * height) - (5.677 * age)
        
    print(f"Hello {name}, your Basal Metabolic Rate (BMR) is: {bmr:.2f} calories/day")
    return bmr
bmr_calculator= bmr_calculator()
maintenance_cal=round( bmr_calculator * 1.2)
loss_cal=round( bmr_calculator * 0.8)
gain_cal=round( bmr_calculator * 1.5)

total_calories = 0
total_carbs=0
totsl_fiber=0
total_protein=0
total_fat=0
while True:
    food_item = input("Enter the food item you consumed (or type 'exit' to finish): ").lower()
    if food_item == 'exit':
        break
    if food_item in food_data:
        quantity = float(input(f"Enter the quantity of {food_item} consumed (in grams): "))
        calories = (food_data[food_item]["calorie"] / 100) * quantity
        carbs = (food_data[food_item]["carbs"] / 100) * quantity
        fiber = (food_data[food_item]["fiber"] / 100) * quantity
        protein = (food_data[food_item]["protein"] / 100) * quantity
        fat = (food_data[food_item]["fat"] / 100) * quantity
        
        total_calories += calories
        total_carbs += carbs
        totsl_fiber += fiber
        total_protein += protein
        total_fat += fat
        
        print(f"{quantity}g of {food_item} contains {calories:.2f} calories, {carbs:.2f}g carbs, {fiber:.2f}g fiber, {protein:.2f}g protein, and {fat:.2f}g fat.")
    else:
        print("Food item not found. Please try again.")
        
        lebels = [ 'Carbs', 'Fiber', 'Protein', 'Fat']
        values = [ total_carbs*4, totsl_fiber*2, total_protein*4, total_fat*9]
        
        plt.pie(values, labels=lebels, autopct='%1.1f%%', startangle=140)
        plt.title('Macronutrient Distribution')
        plt.show()
        
        category = ['maintanace_calories', 'loss_calories', 'gain_calories']
        values = [maintenance_cal, loss_cal, gain_cal]
        plt.figure(figsize=(8, 5))
        bars= plt.bar(category, values, color=['blue', 'orange', 'green'])
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
        plt.title('Caloric Needs')
        plt.ylabel('Calories(kcal)')
        plt.show()
        
        
        
        


        