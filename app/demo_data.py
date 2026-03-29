# app/demo_data.py
import hashlib
from datetime import datetime, timedelta
import random
from app.models import db, User, Employee, Product, Sale, Inventory
import json

def create_demo_data():
    """Create comprehensive demo data"""
    
    # Check if data already exists
    if User.query.first():
        print("Demo data already exists, skipping creation")
        return
    
    print("Creating demo data...")
    
    try:
        # Create users
        users = [
            User(id=1, username='manager', password=hashlib.sha256('manager123'.encode()).hexdigest(), 
                 role='manager', name='John Manager'),
            User(id=2, username='alice', password=hashlib.sha256('emp123'.encode()).hexdigest(), 
                 role='employee', name='Alice Johnson'),
            User(id=3, username='bob', password=hashlib.sha256('emp123'.encode()).hexdigest(), 
                 role='employee', name='Bob Smith'),
            User(id=4, username='carol', password=hashlib.sha256('emp123'.encode()).hexdigest(), 
                 role='employee', name='Carol Davis'),
            User(id=5, username='david', password=hashlib.sha256('emp123'.encode()).hexdigest(), 
                 role='employee', name='David Wilson'),
            User(id=6, username='emma', password=hashlib.sha256('emp123'.encode()).hexdigest(), 
                 role='employee', name='Emma Brown')
        ]
        
        db.session.add_all(users)
        db.session.commit()
        print(f"  - Created {len(users)} users")
        
        # Create employees
        employees = [
            Employee(id=1, user_id=2, name='Alice Johnson', position='Senior Sales Associate', 
                    hire_date=datetime.now() - timedelta(days=540)),
            Employee(id=2, user_id=3, name='Bob Smith', position='Sales Associate', 
                    hire_date=datetime.now() - timedelta(days=365)),
            Employee(id=3, user_id=4, name='Carol Davis', position='Sales Lead', 
                    hire_date=datetime.now() - timedelta(days=720)),
            Employee(id=4, user_id=5, name='David Wilson', position='Sales Associate', 
                    hire_date=datetime.now() - timedelta(days=180)),
            Employee(id=5, user_id=6, name='Emma Brown', position='Sales Associate', 
                    hire_date=datetime.now() - timedelta(days=300))
        ]
        
        db.session.add_all(employees)
        db.session.commit()
        print(f"  - Created {len(employees)} employees")
        
        # Create products
        products_data = [
            {'name': 'MacBook Pro', 'category': 0, 'price': 1299.99, 'stock': 50},
            {'name': 'iPhone 15', 'category': 0, 'price': 999.99, 'stock': 100},
            {'name': 'AirPods Pro', 'category': 0, 'price': 249.99, 'stock': 200},
            {'name': 'iPad Air', 'category': 0, 'price': 599.99, 'stock': 75},
            {'name': 'Apple Watch', 'category': 0, 'price': 399.99, 'stock': 150},
            {'name': 'Leather Jacket', 'category': 1, 'price': 199.99, 'stock': 80},
            {'name': 'Designer Jeans', 'category': 1, 'price': 89.99, 'stock': 120},
            {'name': 'Cotton T-Shirt', 'category': 1, 'price': 29.99, 'stock': 300},
            {'name': 'Winter Coat', 'category': 1, 'price': 149.99, 'stock': 60},
            {'name': 'Running Shoes', 'category': 1, 'price': 79.99, 'stock': 90},
            {'name': 'Organic Apples', 'category': 2, 'price': 4.99, 'stock': 500},
            {'name': 'Fresh Milk', 'category': 2, 'price': 3.49, 'stock': 400},
            {'name': 'Whole Wheat Bread', 'category': 2, 'price': 2.99, 'stock': 300},
            {'name': 'Free Range Eggs', 'category': 2, 'price': 5.99, 'stock': 250},
            {'name': 'Coffee Beans', 'category': 2, 'price': 12.99, 'stock': 150},
            {'name': 'Coffee Table', 'category': 3, 'price': 199.99, 'stock': 40},
            {'name': 'Floor Lamp', 'category': 3, 'price': 89.99, 'stock': 60},
            {'name': 'Wall Art', 'category': 3, 'price': 49.99, 'stock': 100},
            {'name': 'Throw Pillows', 'category': 3, 'price': 24.99, 'stock': 150},
            {'name': 'Area Rug', 'category': 3, 'price': 149.99, 'stock': 30},
            {'name': 'Yoga Mat', 'category': 4, 'price': 29.99, 'stock': 120},
            {'name': 'Dumbbells Set', 'category': 4, 'price': 79.99, 'stock': 80},
            {'name': 'Resistance Bands', 'category': 4, 'price': 19.99, 'stock': 200},
            {'name': 'Exercise Ball', 'category': 4, 'price': 34.99, 'stock': 60},
            {'name': 'Fitness Tracker', 'category': 4, 'price': 59.99, 'stock': 100}
        ]
        
        products = []
        for i, p in enumerate(products_data, 1):
            product = Product(
                id=i,
                name=p['name'],
                category=p['category'],
                price=p['price'],
                stock=p['stock']
            )
            history = generate_product_history(product.name, product.category)
            product.set_history(history)
            products.append(product)
        
        db.session.add_all(products)
        db.session.commit()
        print(f"  - Created {len(products)} products")
        
        # Generate sales data for last 90 days
        sales_data = []
        for day in range(90, -1, -1):
            sale_date = datetime.now() - timedelta(days=day)
            
            for employee in employees:
                # Number of sales varies by employee performance
                if employee.name == 'Alice Johnson':
                    num_sales = random.randint(8, 20)
                elif employee.name == 'Carol Davis':
                    num_sales = random.randint(7, 18)
                elif employee.name == 'Bob Smith':
                    num_sales = random.randint(5, 15)
                elif employee.name == 'Emma Brown':
                    num_sales = random.randint(4, 12)
                else:
                    num_sales = random.randint(3, 10)
                
                for _ in range(num_sales):
                    product = random.choice(products)
                    quantity = random.randint(1, 3)
                    
                    is_weekend = sale_date.weekday() >= 5
                    multiplier = 1.5 if is_weekend else 1.0
                    
                    has_promotion = random.random() < 0.3
                    price_multiplier = 0.9 if has_promotion else 1.0
                    
                    unit_price = product.price * price_multiplier
                    total = unit_price * quantity * multiplier
                    profit = total * 0.35
                    
                    sale = Sale(
                        employee_id=employee.id,
                        product_id=product.id,
                        quantity=quantity,
                        total_amount=total,
                        profit=profit,
                        date=sale_date
                    )
                    sales_data.append(sale)
        
        # Add sales in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(sales_data), batch_size):
            db.session.add_all(sales_data[i:i+batch_size])
            db.session.commit()
        
        print(f"  - Created {len(sales_data)} sales records")
        
        # Update product stock levels
        for product in products:
            product_sales = Sale.query.filter_by(product_id=product.id).all()
            total_sold = sum(s.quantity for s in product_sales)
            product.stock = max(0, product.stock - total_sold)
            update_product_history(product)
        
        db.session.commit()
        
        # Create inventory records
        for product in products:
            inventory = Inventory(
                product_id=product.id,
                stock_level=product.stock,
                reorder_point=random.randint(30, 80),
                optimal_stock=random.randint(150, 300)
            )
            db.session.add(inventory)
        
        db.session.commit()
        print(f"  - Created {len(products)} inventory records")
        
        print(f"\n✅ Demo data created successfully!")
        print(f"\nLogin Credentials:")
        print(f"  Manager: manager / manager123")
        print(f"  Employees: alice / emp123, bob / emp123, carol / emp123, david / emp123, emma / emp123")
        
    except Exception as e:
        print(f"Error creating demo data: {e}")
        db.session.rollback()
        raise

def generate_product_history(product_name, category):
    """Generate realistic historical data for a product"""
    dates = []
    demand = []
    sales = []
    profit = []
    
    for i in range(90, -1, -1):
        date = datetime.now() - timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
        
        category_base = {0: 50, 1: 80, 2: 200, 3: 40, 4: 60}
        base_demand = category_base.get(category, 50)
        
        day_of_week = date.weekday()
        is_weekend = day_of_week >= 5
        weekday_multiplier = 1.5 if is_weekend else 0.8
        
        trend = 1 + (90 - i) / 300
        variation = random.uniform(0.7, 1.3)
        
        daily_demand = base_demand * weekday_multiplier * trend * variation
        daily_sales = daily_demand * base_demand * 2
        daily_profit = daily_sales * 0.35
        
        demand.append(max(1, int(daily_demand)))
        sales.append(max(10, daily_sales))
        profit.append(max(1, daily_profit))
    
    return {
        'dates': dates,
        'demand': demand,
        'sales': sales,
        'profit': profit
    }

def update_product_history(product):
    """Update product history with actual sales data"""
    sales = Sale.query.filter_by(product_id=product.id).order_by(Sale.date).all()
    
    daily_data = {}
    for sale in sales:
        date_str = sale.date.strftime('%Y-%m-%d')
        if date_str not in daily_data:
            daily_data[date_str] = {'demand': 0, 'sales': 0, 'profit': 0}
        daily_data[date_str]['demand'] += sale.quantity
        daily_data[date_str]['sales'] += sale.total_amount
        daily_data[date_str]['profit'] += sale.profit
    
    dates = []
    demand = []
    sales_list = []
    profit_list = []
    
    for date_str in sorted(daily_data.keys()):
        dates.append(date_str)
        demand.append(daily_data[date_str]['demand'])
        sales_list.append(daily_data[date_str]['sales'])
        profit_list.append(daily_data[date_str]['profit'])
    
    product.set_history({
        'dates': dates,
        'demand': demand,
        'sales': sales_list,
        'profit': profit_list
    })