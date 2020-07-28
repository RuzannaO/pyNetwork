import re
from ..models import Meal, Ingredient, Category, Area, Meal_ingredient
# from ..app.users.models import User_Favorite, User_Favorite_Category, User


def name_correct(name):
    matches = re.finditer(" ", name)
    list1 = [match.start() for match in matches]
    name1 = name[0]
    for i in range(0, len(name) - 1):
        if i in list1:
            name1 += name[i] + name[i + 1].upper()
        else:
            name1 += name[i + 1].lower()
    name1 = name1.replace("  ", " ")
    return (name1)

def fav_category(meallist,user_categories):
    meal_list=[]
    for i in meallist:
        print(i.id,i.category_id)
        meal_list.append([i.id,i.category_id])
        # user_categories=User_Favorite_Category.query.all()
    us_cat=[]
    for x in user_categories:
        us_cat.append([x.user_id,x.category_id])
    l1=[]
    for i in us_cat:
        l1.append(i[0])
    user_dict = {key: [] for key in l1}
    for x in us_cat:
        user_dict[x[0]].append(x[1])
    for k in user_dict.keys():
        for j in user_dict[k]:
            for c in meal_list:
                if j == c[1]:
                    print(k,c[0])
                    # print("user:",User.query.filter_by(id=k).first().username)
                    # print("email:",User.query.filter_by(id=k).first().email)
                    print("meal name:",Meal.query.filter_by(id=c[0]).first().name)
                    print("category name:", Category.query.filter_by(id=c[1]).first().name)

