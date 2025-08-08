import psycopg2

def get_database_connection():
    return psycopg2.connect(
        host='localhost',
        port=5432,
        user='postgres',
        password='Zhengnan4568',
        database='register'
    )

def init_file_table():
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id SERIAL PRIMARY KEY,
            file_path VARCHAR(255) NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_file_table()

