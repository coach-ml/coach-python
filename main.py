import coach

if __name__ == "__main__":
    coach = Coach('api-key')
    results = coach.predict('model', 'rose.jpg')
    
    print(results)