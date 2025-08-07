import csv
import re
from typing import List, Dict, Tuple

class BusinessIdeaGenerator:
    def __init__(self):
        """
        initialize the business idea generator with csv data
        """
        self.csv_file_path = ".vscode/business_idea.csv"
        self.business_data = []
        self.load_business_data()
    
    def load_business_data(self):
        """
        load business data from csv file
        """
        try:
            with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    business_info = {
                        'id': int(row['business_id']),
                        'idea': row['business_idea'].lower().strip(),
                        'category': row['category'].lower().strip(),
                        'skills': [
                            row['skill_1'].lower().strip(),
                            row['skill_2'].lower().strip(),
                            row['skill_3'].lower().strip(),
                            row['skill_4'].lower().strip(),
                            row['skill_5'].lower().strip()
                        ]
                    }
                    self.business_data.append(business_info)
            print(f"successfully loaded {len(self.business_data)} business ideas from csv file")
        except FileNotFoundError:
            print(f"error: csv file '{self.csv_file_path}' not found")
            print("please make sure the csv file is in the 'vscode' folder")
        except Exception as e:
            print(f"error loading csv file: {e}")
    
    def normalize_skill(self, skill: str) -> str:
        """
        normalize skill text for better matching
        """
        skill = skill.lower().strip()
        skill = re.sub(r'[^\w\s\-\(\)\/\&]', ' ', skill)
        skill = re.sub(r'\s+', ' ', skill)
        return skill
    
    def calculate_skill_match(self, user_skills: List[str], business_skills: List[str]) -> Tuple[int, List[str]]:
        """
        calculate how many skills match between user and business
        returns tuple of (match_count, matched_skills_list)
        """
        normalized_user_skills = [self.normalize_skill(skill) for skill in user_skills]
        normalized_business_skills = [self.normalize_skill(skill) for skill in business_skills]
        
        matches = []
        match_count = 0
        
        for user_skill in normalized_user_skills:
            for business_skill in normalized_business_skills:
                if user_skill == business_skill:
                    matches.append(user_skill)
                    match_count += 1
                    break
                elif (len(user_skill) > 3 and user_skill in business_skill) or \
                     (len(business_skill) > 3 and business_skill in user_skill):
                    matches.append(f"{user_skill} (similar to: {business_skill})")
                    match_count += 1
                    break
                elif len(user_skill.split()) > 1 and len(business_skill.split()) > 1:
                    user_words = set(user_skill.split())
                    business_words = set(business_skill.split())
                    common_words = user_words.intersection(business_words)
                    if len(common_words) >= 2:
                        matches.append(f"{user_skill} (keywords match: {business_skill})")
                        match_count += 1
                        break
        
        return match_count, matches
    
    def get_user_skills(self) -> List[str]:
        """
        get 5 skills from user input
        """
        print("\n" + "="*60)
        print("BUSINESS IDEA GENERATOR")
        print("="*60)
        print("please enter your 5 skills (press enter after each skill):")
        print("tip: be specific (e.g., 'python programming', 'social media marketing')")
        print("-"*60)
        
        skills = []
        for i in range(5):
            while True:
                skill = input(f"skill {i+1}: ").strip()
                if skill:
                    skills.append(skill.lower())
                    break
                else:
                    print("please enter a valid skill")
        
        return skills
    
    def find_matching_businesses(self, user_skills: List[str], min_matches: int = 3) -> List[Dict]:
        """
        find businesses that match at least min_matches skills
        """
        matching_businesses = []
        
        for business in self.business_data:
            match_count, matched_skills = self.calculate_skill_match(user_skills, business['skills'])
            
            if match_count >= min_matches:
                business_match = {
                    'business': business,
                    'match_count': match_count,
                    'matched_skills': matched_skills,
                    'match_percentage': (match_count / 5) * 100
                }
                matching_businesses.append(business_match)
        
        matching_businesses.sort(key=lambda x: (x['match_count'], x['match_percentage']), reverse=True)
        
        return matching_businesses
    
    def display_results(self, user_skills: List[str], matching_businesses: List[Dict]):
        """
        display the matching business ideas to user
        """
        print("\n" + "="*80)
        print("YOUR SKILLS:")
        print("="*80)
        for i, skill in enumerate(user_skills, 1):
            print(f"{i}. {skill.title()}")
        
        print("\n" + "="*80)
        print("RECOMMENDED BUSINESS IDEAS:")
        print("="*80)
        
        if not matching_businesses:
            print("sorry, no business ideas found that match at least 3 of your skills.")
            print("try entering more general or common skills.")
            return
        
        for i, match in enumerate(matching_businesses, 1):
            business = match['business']
            print(f"\n{i}. {business['idea'].title()}")
            print(f"   category: {business['category'].title()}")
            print(f"   match strength: {match['match_count']}/5 skills ({match['match_percentage']:.1f}%)")
            print(f"   your matching skills:")
            for skill in match['matched_skills']:
                print(f"      • {skill}")
            
            print(f"   all required skills for this business:")
            for j, skill in enumerate(business['skills'], 1):
                print(f"      {j}. {skill}")
            print("-" * 60)
    
    def run_generator(self):
        """
        main method to run the business idea generator
        """
        if not self.business_data:
            print("cannot run generator: no business data loaded")
            return
        
        while True:
            try:
                user_skills = self.get_user_skills()
                matching_businesses = self.find_matching_businesses(user_skills, min_matches=3)
                self.display_results(user_skills, matching_businesses)
                
                print("\n" + "="*60)
                choice = input("would you like to try with different skills? (y/n): ").lower().strip()
                if choice != 'y' and choice != 'yes':
                    break
                    
            except KeyboardInterrupt:
                print("\n\nprogram interrupted by user. goodbye!")
                break
            except Exception as e:
                print(f"an error occurred: {e}")
                choice = input("would you like to try again? (y/n): ").lower().strip()
                if choice != 'y' and choice != 'yes':
                    break
        
        print("\nthank you for using the business idea generator!")
        print("good luck with your entrepreneurial journey! 🚀")

def main():
    """
    main function to run the program
    """
    print("starting business idea generator...")
    print("loading data from: vscode/business_idea.csv")
    
    generator = BusinessIdeaGenerator()
    generator.run_generator()

if __name__ == "__main__":
    main()