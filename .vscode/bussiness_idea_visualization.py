import csv
import re
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import seaborn as sns
from datetime import datetime

@dataclass
class BusinessAnalytics:
    """Data class to store business analytics"""
    success_rate: float
    risk_level: str
    market_demand: float
    competition_level: str
    growth_potential: float
    failure_rate: float

class EnhancedBusinessIdeaGenerator:
    def _init_(self, csv_file_path: str):
        """
        Initialize the enhanced business idea generator with csv data
        """
        self.csv_file_path = csv_file_path
        self.business_data = []
        self.load_business_data()
        self.setup_visualization_style()
    
    def setup_visualization_style(self):
        """Setup matplotlib and seaborn styling"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def load_business_data(self):
        """Load business data from enhanced csv file"""
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
                        ],
                        'startup_capital_min': int(row['startup_capital_min']),
                        'startup_capital_max': int(row['startup_capital_max']),
                        'capital_category': row['capital_category'].lower().strip()
                    }
                    self.business_data.append(business_info)
            print(f"✅ Successfully loaded {len(self.business_data)} business ideas from csv file")
        except FileNotFoundError:
            print(f"❌ Error: csv file '{self.csv_file_path}' not found")
            print("Please make sure the csv file is in the same directory as this program")
        except Exception as e:
            print(f"❌ Error loading csv file: {e}")
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize skill text for better matching"""
        skill = skill.lower().strip()
        skill = re.sub(r'[^\w\s\-\(\)\/\&]', ' ', skill)
        skill = re.sub(r'\s+', ' ', skill)
        return skill
    
    def calculate_skill_match(self, user_skills: List[str], business_skills: List[str]) -> Tuple[int, List[str]]:
        """Calculate how many skills match between user and business"""
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
        """Get 5 skills from user input"""
        print("\n" + "="*80)
        print("🚀 ENHANCED BUSINESS IDEA GENERATOR WITH ANALYTICS")
        print("="*80)
        print("Please enter your 5 skills (press enter after each skill):")
        print("💡 Tip: Be specific (e.g., 'python programming', 'social media marketing')")
        print("-"*80)
        
        skills = []
        for i in range(5):
            while True:
                skill = input(f"Skill {i+1}: ").strip()
                if skill:
                    skills.append(skill.lower())
                    break
                else:
                    print("Please enter a valid skill")
        
        return skills
    
    def find_matching_businesses(self, user_skills: List[str], min_matches: int = 3) -> List[Dict]:
        """Find businesses that match at least min_matches skills"""
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
    
    def display_business_options(self, matching_businesses: List[Dict]) -> Dict:
        """Display business options and let user select one"""
        print("\n" + "="*80)
        print("📊 YOUR MATCHING BUSINESS IDEAS:")
        print("="*80)
        
        if not matching_businesses:
            print("❌ Sorry, no business ideas found that match at least 3 of your skills.")
            return None
        
        # Display top 10 matches
        display_count = min(10, len(matching_businesses))
        
        for i in range(display_count):
            match = matching_businesses[i]
            business = match['business']
            print(f"\n{i+1}. {business['idea'].title()}")
            print(f"   📂 Category: {business['category'].title()}")
            print(f"   ⭐ Match: {match['match_count']}/5 skills ({match['match_percentage']:.1f}%)")
            print(f"   💰 Capital: ${business['startup_capital_min']:,} - ${business['startup_capital_max']:,} ({business['capital_category']})")
            print(f"   🎯 Matching skills: {', '.join(match['matched_skills'][:2])}{'...' if len(match['matched_skills']) > 2 else ''}")
        
        # Get user selection
        while True:
            try:
                print("\n" + "-"*60)
                choice = input(f"Select a business idea (1-{display_count}) for detailed analysis: ").strip()
                choice_num = int(choice)
                if 1 <= choice_num <= display_count:
                    return matching_businesses[choice_num - 1]
                else:
                    print(f"Please enter a number between 1 and {display_count}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                return None
    
    def simulate_business_analytics(self, business_idea: str, category: str, capital_range: Tuple[int, int]) -> BusinessAnalytics:
        """
        Simulate business analytics data (in a real implementation, this would use web scraping/APIs)
        This creates realistic-looking data based on business type and capital requirements
        """
        print("🔍 Analyzing market data and business metrics...")
        time.sleep(2)  # Simulate API call delay
        
        # Base metrics by category
        category_metrics = {
            'technology & software': {'base_success': 0.65, 'base_risk': 'medium-high', 'growth': 0.85},
            'e-commerce & retail': {'base_success': 0.55, 'base_risk': 'medium', 'growth': 0.70},
            'food & beverage': {'base_success': 0.45, 'base_risk': 'high', 'growth': 0.60},
            'health & wellness': {'base_success': 0.70, 'base_risk': 'medium', 'growth': 0.75},
            'creative services': {'base_success': 0.60, 'base_risk': 'medium', 'growth': 0.65},
            'education & training': {'base_success': 0.75, 'base_risk': 'low-medium', 'growth': 0.80},
            'home & lifestyle': {'base_success': 0.65, 'base_risk': 'medium', 'growth': 0.55},
            'transportation & logistics': {'base_success': 0.50, 'base_risk': 'medium-high', 'growth': 0.70},
            'finance & consulting': {'base_success': 0.70, 'base_risk': 'medium', 'growth': 0.75},
            'automotive & mechanical': {'base_success': 0.60, 'base_risk': 'medium', 'growth': 0.50},
            'real estate & property': {'base_success': 0.55, 'base_risk': 'high', 'growth': 0.65},
            'agriculture & sustainability': {'base_success': 0.65, 'base_risk': 'medium-high', 'growth': 0.80},
            'entertainment & events': {'base_success': 0.45, 'base_risk': 'high', 'growth': 0.60},
            'pet & animal services': {'base_success': 0.70, 'base_risk': 'low-medium', 'growth': 0.65},
            'manufacturing & crafts': {'base_success': 0.55, 'base_risk': 'medium-high', 'growth': 0.55},
            'sports & recreation': {'base_success': 0.60, 'base_risk': 'medium', 'growth': 0.65},
            'beauty & personal care': {'base_success': 0.65, 'base_risk': 'medium', 'growth': 0.60},
            'senior & childcare services': {'base_success': 0.75, 'base_risk': 'low', 'growth': 0.70},
            'specialty services': {'base_success': 0.68, 'base_risk': 'medium', 'growth': 0.65}
        }
        
        base_metrics = category_metrics.get(category, {'base_success': 0.60, 'base_risk': 'medium', 'growth': 0.65})
        
        # Adjust based on capital requirements
        capital_avg = (capital_range[0] + capital_range[1]) / 2
        capital_factor = min(1.0, capital_avg / 100000)  # Normalize to 100k
        
        # Higher capital often means lower risk but also higher barriers
        success_rate = base_metrics['base_success'] * (0.9 + 0.2 * capital_factor) + np.random.normal(0, 0.05)
        success_rate = max(0.1, min(0.95, success_rate))
        
        # Market demand simulation
        market_demand = 0.6 + np.random.normal(0, 0.15)
        market_demand = max(0.2, min(0.9, market_demand))
        
        # Growth potential
        growth_potential = base_metrics['growth'] + np.random.normal(0, 0.1)
        growth_potential = max(0.3, min(0.95, growth_potential))
        
        # Competition level based on entry barriers (capital)
        if capital_avg < 10000:
            competition_level = 'high'
        elif capital_avg < 50000:
            competition_level = 'medium'
        else:
            competition_level = 'low-medium'
        
        return BusinessAnalytics(
            success_rate=success_rate,
            risk_level=base_metrics['base_risk'],
            market_demand=market_demand,
            competition_level=competition_level,
            growth_potential=growth_potential,
            failure_rate=1 - success_rate
        )
    
    def create_business_analytics_visualization(self, business: Dict, analytics: BusinessAnalytics):
        """Create comprehensive visualization of business analytics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Business Analytics: {business["business"]["idea"].title()}', fontsize=16, fontweight='bold')
        
        # 1. Success Rate Gauge Chart
        self.create_gauge_chart(ax1, analytics.success_rate, "Success Rate", "green")
        
        # 2. Risk vs Reward Scatter
        self.create_risk_reward_chart(ax2, business, analytics)
        
        # 3. Market Metrics Bar Chart
        self.create_market_metrics_chart(ax3, analytics)
        
        # 4. Financial Overview
        self.create_financial_overview(ax4, business, analytics)
        
        plt.tight_layout()
        plt.show()
        
        # Create additional detailed chart
        self.create_detailed_analysis_chart(business, analytics)
    
    def create_gauge_chart(self, ax, value, title, color):
        """Create a gauge chart for success rate"""
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background arc
        ax.plot(theta, r, 'lightgray', linewidth=10)
        
        # Value arc
        value_theta = np.linspace(0, np.pi * value, int(100 * value))
        value_r = np.ones_like(value_theta)
        ax.plot(value_theta, value_r, color, linewidth=10)
        
        # Add percentage text
        ax.text(0, 0, f'{value:.1%}', ha='center', va='center', fontsize=20, fontweight='bold')
        ax.text(0, -0.3, title, ha='center', va='center', fontsize=12)
        
        ax.set_ylim(-0.5, 1.1)
        ax.set_xlim(-0.2, np.pi + 0.2)
        ax.axis('off')
    
    def create_risk_reward_chart(self, ax, business, analytics):
        """Create risk vs reward scatter plot"""
        # Map risk levels to numeric values
        risk_mapping = {'low': 1, 'low-medium': 2, 'medium': 3, 'medium-high': 4, 'high': 5}
        risk_numeric = risk_mapping.get(analytics.risk_level, 3)
        
        # Calculate reward (inverse of risk, adjusted by success rate)
        reward = (6 - risk_numeric) * analytics.success_rate * 2
        
        # Plot the business
        ax.scatter([risk_numeric], [reward], s=300, alpha=0.7, c='red')
        ax.annotate(business['business']['idea'].title(), 
                   (risk_numeric, reward), 
                   xytext=(10, 10), textcoords='offset points', fontsize=10)
        
        # Add reference businesses for context
        reference_points = [
            (1, 8, 'Low Risk\nHigh Reward'),
            (5, 2, 'High Risk\nLow Reward'),
            (3, 5, 'Balanced')
        ]
        
        for x, y, label in reference_points:
            ax.scatter([x], [y], s=100, alpha=0.3, c='gray')
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Potential Reward')
        ax.set_title('Risk vs Reward Analysis')
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
        ax.grid(True, alpha=0.3)
    
    def create_market_metrics_chart(self, ax, analytics):
        """Create market metrics bar chart"""
        metrics = {
            'Market Demand': analytics.market_demand,
            'Growth Potential': analytics.growth_potential,
            'Success Rate': analytics.success_rate
        }
        
        bars = ax.bar(metrics.keys(), metrics.values(), 
                     color=['skyblue', 'lightgreen', 'gold'], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.1%}', ha='center', va='bottom')
        
        ax.set_ylabel('Percentage')
        ax.set_title('Market Metrics')
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def create_financial_overview(self, ax, business, analytics):
        """Create financial overview chart"""
        capital_min = business['business']['startup_capital_min']
        capital_max = business['business']['startup_capital_max']
        capital_avg = (capital_min + capital_max) / 2
        
        # Estimate potential returns based on success rate and market demand
        potential_return_low = capital_avg * analytics.success_rate * 0.5
        potential_return_high = capital_avg * analytics.success_rate * analytics.market_demand * 3
       
        categories = ['Initial\nInvestment', 'Potential\nReturn (Low)', 'Potential\nReturn (High)']
        values = [capital_avg, potential_return_low, potential_return_high]
        colors = ['red', 'orange', 'green']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${value:,.0f}', ha='center', va='bottom')
        
        ax.set_ylabel('Amount (USD)')
        ax.set_title('Financial Overview')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def create_detailed_analysis_chart(self, business, analytics):
        """Create a detailed analysis chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Detailed Analysis: {business["business"]["idea"].title()}', fontsize=14, fontweight='bold')
        
        # Monthly projection over 2 years
        months = np.arange(1, 25)
        
        # Simulate revenue growth (S-curve)
        base_revenue = (business['business']['startup_capital_min'] + business['business']['startup_capital_max']) / 2 * 0.1
        revenue_projection = base_revenue * analytics.success_rate * (1 - np.exp(-months/6)) * analytics.growth_potential
        
        # Add some realistic variation
        revenue_projection *= (1 + np.random.normal(0, 0.1, len(months)))
        revenue_projection = np.maximum(0, revenue_projection)  # Ensure non-negative
        
        ax1.plot(months, revenue_projection, 'b-', linewidth=2, label='Projected Revenue')
        ax1.fill_between(months, revenue_projection * 0.8, revenue_projection * 1.2, alpha=0.2)
        ax1.set_xlabel('Months')
        ax1.set_ylabel('Monthly Revenue ($)')
        ax1.set_title('24-Month Revenue Projection')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Competition and Market Position
        competition_data = {
            'Market Share\nPotential': analytics.market_demand * 100,
            'Competition\nIntensity': {'low': 20, 'low-medium': 40, 'medium': 60, 'medium-high': 80, 'high': 100}.get(analytics.competition_level, 60),
            'Entry Barrier\nHeight': min(100, (business['business']['startup_capital_max'] / 10000) * 10)
        }
        
        ax2.bar(competition_data.keys(), competition_data.values(), 
               color=['green', 'orange', 'red'], alpha=0.7)
        ax2.set_ylabel('Percentage/Score')
        ax2.set_title('Market Position Analysis')
        
        # Add value labels
        for i, (key, value) in enumerate(competition_data.items()):
            ax2.text(i, value + 2, f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def display_detailed_business_info(self, selected_business: Dict, analytics: BusinessAnalytics):
        """Display detailed information about the selected business"""
        business = selected_business['business']
        
        print("\n" + "="*80)
        print(f"📋 DETAILED BUSINESS ANALYSIS")
        print("="*80)
        print(f"🏢 Business: {business['idea'].title()}")
        print(f"📂 Category: {business['category'].title()}")
        print(f"💰 Capital Range: ${business['startup_capital_min']:,} - ${business['startup_capital_max']:,}")
        print(f"💎 Capital Category: {business['capital_category'].title()}")
        
        print(f"\n📊 MARKET ANALYTICS:")
        print("-"*40)
        print(f"✅ Success Rate: {analytics.success_rate:.1%}")
        print(f"⚠  Risk Level: {analytics.risk_level.title()}")
        print(f"📈 Market Demand: {analytics.market_demand:.1%}")
        print(f"🏆 Competition Level: {analytics.competition_level.title()}")
        print(f"🚀 Growth Potential: {analytics.growth_potential:.1%}")
        print(f"❌ Failure Rate: {analytics.failure_rate:.1%}")
        
        print(f"\n🎯 YOUR SKILL MATCH:")
        print("-"*40)
        print(f"Match Strength: {selected_business['match_count']}/5 skills ({selected_business['match_percentage']:.1f}%)")
        print("Matched Skills:")
        for skill in selected_business['matched_skills']:
            print(f"  ✓ {skill}")
        
        print(f"\n📝 ALL REQUIRED SKILLS:")
        print("-"*40)
        for i, skill in enumerate(business['skills'], 1):
            status = "✓" if any(s.split(' (')[0] in skill for s in selected_business['matched_skills']) else "○"
            print(f"  {status} {i}. {skill.title()}")
        
        # Investment recommendation
        print(f"\n💡 INVESTMENT RECOMMENDATION:")
        print("-"*40)
        if analytics.success_rate > 0.7:
            recommendation = "🟢 HIGHLY RECOMMENDED"
        elif analytics.success_rate > 0.5:
            recommendation = "🟡 MODERATELY RECOMMENDED"
        else:
            recommendation = "🔴 HIGH RISK - CONSIDER CAREFULLY"
        
        print(f"Status: {recommendation}")
        
        capital_avg = (business['startup_capital_min'] + business['startup_capital_max']) / 2
        roi_estimate = capital_avg * analytics.success_rate * analytics.market_demand * 2
        print(f"Estimated 2-year ROI: ${roi_estimate:,.0f}")
        
        payback_months = 24 / max(analytics.success_rate * analytics.market_demand * 2, 0.1)
        print(f"Estimated Payback Period: {payback_months:.1f} months")
    
    def run_enhanced_generator(self):
        """Main method to run the enhanced business idea generator"""
        if not self.business_data:
            print("❌ Cannot run generator: no business data loaded")
            return
        
        while True:
            try:
                # Get user skills
                user_skills = self.get_user_skills()
                
                # Find matching businesses
                matching_businesses = self.find_matching_businesses(user_skills, min_matches=3)
                
                if not matching_businesses:
                    print("\n❌ No business ideas found that match at least 3 of your skills.")
                    print("💡 Try entering more general or common skills.")
                    choice = input("\nWould you like to try with different skills? (y/n): ").lower().strip()
                    if choice != 'y' and choice != 'yes':
                        break
                    continue
                
                # Display options and get user selection
                selected_business = self.display_business_options(matching_businesses)
                
                if selected_business is None:
                    print("\n👋 Thanks for using the Business Idea Generator!")
                    break
                
                # Get analytics for selected business
                business = selected_business['business']
                print(f"\n🔄 Analyzing '{business['idea'].title()}'...")
                
                analytics = self.simulate_business_analytics(
                    business['idea'], 
                    business['category'],
                    (business['startup_capital_min'], business['startup_capital_max'])
                )
                
                # Display detailed information
                self.display_detailed_business_info(selected_business, analytics)
                
                # Ask if user wants to see visualization
                show_viz = input("\n📊 Would you like to see visual analytics? (y/n): ").lower().strip()
                if show_viz == 'y' or show_viz == 'yes':
                    self.create_business_analytics_visualization(selected_business, analytics)
                
                # Ask if user wants to continue
                print("\n" + "="*60)
                choice = input("🔄 Would you like to analyze another business idea? (y/n): ").lower().strip()
                if choice != 'y' and choice != 'yes':
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Program interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                choice = input("Would you like to try again? (y/n): ").lower().strip()
                if choice != 'y' and choice != 'yes':
                    break
        
        print("\n🎉 Thank you for using the Enhanced Business Idea Generator!")
        print("📈 Good luck with your entrepreneurial journey! 🚀")

def main():
    """Main function to run the program"""
    csv_filename =".vscode/business_id,business_idea.csv"   # Update this to your enhanced CSV file name
    
    print("🚀 Starting Enhanced Business Idea Generator...")
    print(f"📁 Loading data from: {csv_filename}")
    
    # Check if required packages are installed
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install required packages:")
        print("pip install matplotlib seaborn numpy")
        return
    
    generator = EnhancedBusinessIdeaGenerator(csv_filename)
    generator.run_enhanced_generator()

if _name_ == "_main_":
    main()