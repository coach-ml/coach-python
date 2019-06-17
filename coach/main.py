from coach import Coach

if __name__ == "__main__":
    #coach = Coach().login('B1dp8OKfOk2RskZ9EkJai3rIAkENZQky2eINkWhi')
    #coach.cache_model('flowers')
    
    coach = Coach()
    result = coach.get_model('flowers').predict('rose.jpg')

    #o = open('rose.jpg', 'rb')
    #rose = o.read()
    #o.close()
    #result = coach.get_model('flowers').predict(rose)

    print(result)