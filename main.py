from app import app

# Gets the app from app.py and runs it
# app.run()
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)