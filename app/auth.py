import sys
import os
import json
import hashlib

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from flask import Blueprint, render_template, redirect, url_for, request, flash, session
from functools import wraps

auth_bp = Blueprint('auth', __name__)

# Simple user database (in production, use a real database)
USERS_FILE = os.path.join(parent_dir, 'users.json')

def init_users():
    """Initialize users database"""
    if not os.path.exists(USERS_FILE):
        users = {
            'manager': {
                'password': hashlib.sha256('manager123'.encode()).hexdigest(),
                'role': 'manager',
                'name': 'Store Manager'
            },
            'employee1': {
                'password': hashlib.sha256('emp123'.encode()).hexdigest(),
                'role': 'employee',
                'name': 'John Doe',
                'employee_id': 'EMP_001'
            },
            'employee2': {
                'password': hashlib.sha256('emp123'.encode()).hexdigest(),
                'role': 'employee',
                'name': 'Jane Smith',
                'employee_id': 'EMP_002'
            }
        }
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)

def get_user(username):
    """Get user from database"""
    if not os.path.exists(USERS_FILE):
        init_users()
    
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    return users.get(username)

def login_required(role=None):
    """Decorator to check if user is logged in and has required role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                flash('Please login first', 'warning')
                return redirect(url_for('auth.login'))
            
            if role and session['user']['role'] != role:
                flash('Access denied. Insufficient permissions.', 'danger')
                return redirect(url_for('auth.dashboard_redirect'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = get_user(username)
        
        if user and user['password'] == hashlib.sha256(password.encode()).hexdigest():
            session['user'] = {
                'username': username,
                'role': user['role'],
                'name': user['name']
            }
            
            if user.get('employee_id'):
                session['user']['employee_id'] = user['employee_id']
            
            flash(f'Welcome back, {user["name"]}!', 'success')
            
            # Redirect based on role
            if user['role'] == 'manager':
                return redirect(url_for('routes.manager_dashboard'))
            else:
                return redirect(url_for('routes.employee_dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/dashboard-redirect')
def dashboard_redirect():
    """Redirect to appropriate dashboard"""
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    
    if session['user']['role'] == 'manager':
        return redirect(url_for('routes.manager_dashboard'))
    else:
        return redirect(url_for('routes.employee_dashboard'))

# Initialize users on import
init_users()