from flask import (
    Blueprint,
    url_for,
    redirect,
    render_template, request, Flask
)

import os
satatic_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))
from .models import Meal, Ingredient, Category, Area, Meal_ingredient
from .. import db
from ..users.models import User_Favorite, User_Favorite_Category, User
from flask_paginate import Pagination, get_page_parameter
import re
from .send_msg import Mail,app, send_mail
from flask_mail import Mail, Message
from .lib.funct import name_correct, fav_category



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


from ..users.forms import CommentForm
from ..users.models import UserComments
from flask_login import current_user

meals_bp = Blueprint(
    'meals',
    __name__,
    template_folder='templates',
    url_prefix='/meal',
    static_folder='static',
    static_url_path=satatic_path
)


@meals_bp.route('/fill_db')
def fill_db():
    # before_run_last=Meal.query.all()
    from .fill_db import fill_all
    fill_all()
    # after_run_last = Meal.query.all()
    # meallist = Meal.query.filter(Meal.id > before_run_last[-1].id).all()
    # user_categories=User_Favorite_Category.query.all()
    # fav_category(meallist,user_categories)
    # return f' before run {before_run_last[-1].id} and after run {after_run_last[-1].id}'
    return 'Updated'


@meals_bp.route('/')
def test_route():
    page = request.args.get('page', 1, type=int)
    meallist = Meal.query.paginate(page, 6, False)
    next_url = url_for('meals.test_route', page=meallist.next_num) \
        if meallist.has_next else None
    prev_url = url_for('meals.test_route', page=meallist.prev_num) \
        if meallist.has_prev else None

    return render_template(

        'main_page.html',
        meallist=meallist,
        next_url=next_url,
        prev_url=prev_url,

    )


@meals_bp.route('/categories')
def categories_list():
    categories = Category.query.all()
    return render_template('category_list.html', categories=categories)


@meals_bp.route('/search/<meal_name>/')
def meal_search(meal_name):
    meal_name = name_correct(meal_name)
    meal = Meal.query.filter_by(name=meal_name).first()
    if meal:
        form = CommentForm()
        check = False
        page = request.args.get('page', 1, type=int)
        comments = UserComments.query.filter(UserComments.meal_id == meal.id).paginate(per_page=2, page=page)
        ingredients = Meal_ingredient.query.filter_by(meal_id=meal.id).all()
        return render_template('meal_info.html', meal=meal, ingredients=ingredients,
                           form=form,check = check,comments=comments,page=page)
    else:

        return redirect('/meal')


@meals_bp.route('/category/<int:c_id>/')
def meals_by_category(c_id):
    page = request.args.get('page', 1, type=int)

    meallist = Meal.query.filter_by(category_id=c_id).paginate(page, 6, False)
    next_url = url_for('meals.meals_by_category', c_id=c_id, page=meallist.next_num) \
        if meallist.has_next else None
    prev_url = url_for('meals.meals_by_category', c_id=c_id, page=meallist.prev_num) \
        if meallist.has_prev else None

    return render_template(

        'main_page.html',
        meallist=meallist,
        next_url=next_url,
        prev_url=prev_url,

    )


@meals_bp.route('/add-favorite/<int:meal_id>', methods=['GET'])
# @login_required
def add_favorite(meal_id):
    check = False

    bb = db.session.query(User_Favorite).filter(
        User_Favorite.meal_id.like(meal_id),
        User_Favorite.user_id.like(current_user.id)).first()

    if not bb:
        check = False
        user_favorite = User_Favorite(
            user_id=current_user.id,
            meal_id=meal_id
        )

        db.session.add(user_favorite)
        db.session.commit()
    else:
        check = True
        db.session.delete(bb)
        db.session.commit()
    return redirect(url_for('meals.meal_info', m_id=meal_id))

# ################################
@meals_bp.route('/add-favorite-category/<int:category_id>', methods=['GET'])
# @login_required
def add_favorite_category(category_id):
    # Check = False

    bb = db.session.query(User_Favorite_Category).filter(
        User_Favorite_Category.category_id.like(category_id),
        User_Favorite_Category.user_id.like(current_user.id)).first()

    if not bb:
        check = False
        user_favorite_category = User_Favorite_Category(
            user_id=current_user.id,
            category_id=category_id
        )

        db.session.add(user_favorite_category)
        db.session.commit()
        print(check)
    else:
        check = True
        db.session.delete(bb)
        db.session.commit()
        print(check)
    return redirect(url_for('meals.categories_list'))

###########################################













@meals_bp.route('/meal_info/<int:m_id>/', methods=['GET'])
def meal_info(m_id):
    meal = Meal.query.filter_by(id=m_id).first()
    ingredients = Meal_ingredient.query.filter_by(meal_id=m_id).all()
    form = CommentForm()
    check = db.session.query(User_Favorite).filter(
        User_Favorite.meal_id.like(m_id),
        User_Favorite.user_id.like(current_user.id)
    ).first()

    page = request.args.get('page', 1, type=int)
    comments = UserComments.query.filter(UserComments.meal_id == m_id).paginate(per_page=2, page=page)
    return render_template('meal_info.html',
                           meal=meal, ingredients=ingredients, check=check,
                           form=form, comments=comments, page=page, m_id=m_id)


@meals_bp.route('/meal_info/<int:m_id>/', methods=['POST'])
def meal_info_post(m_id):
    form = CommentForm()
    if form.is_submitted():
        comment = UserComments(content=form.content.data, user_id=current_user.id, meal_id=m_id)
        db.session.add(comment)
        db.session.commit()
        return redirect(url_for('meals.meal_info',
                                m_id=m_id))
