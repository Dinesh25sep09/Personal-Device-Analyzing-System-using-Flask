from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from data import Articles
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
import pickle
import seaborn as sns
import pandas as pd
import pymysql
import io
from PIL import Image
import base64
import cv2
import numpy as np
import glob
import  sys
from PIL import Image
from logging import FileHandler,WARNING
import os
from os import listdir
from os.path import isfile, join
from itsdangerous import base64_encode, base64_decode



app = Flask(__name__)
# Config MySQL

m="data/mobile.pickle"
s=pickle.load(open(m,"rb"))



m1="data/laptop.pickle"
s1=pickle.load(open(m1,"rb"))

m2="data/tablet.pickle"
s2=pickle.load(open(m2,"rb"))

m3="data/camera.pickle"
s3=pickle.load(open(m3,"rb"))






app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'admin'
app.config['MYSQL_DB'] = 'personaldevice'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# init MYSQL
mysql = MySQL(app)


Articles = Articles()



@app.route('/')
def index():
        return render_template('home.html')

@app.route('/about')
def about():
        return render_template('about.html')


@app.route('/services')
def service():
        return render_template('services.html')


@app.route('/mobile')
def hello():
        return render_template("mobile.html")

def is_log_in(f):
     @wraps(f)
     def wrap(*args, **kwargs):
          if 'log' in session:
               return f(*args,**kwargs)
          else:
               flash('Unauthorized, Please Login','danger')
               return redirect(url_for('login'))
     return wrap 





@app.route('/image', methods=['GET', 'POST'])
def image():
        if request.method=='POST':
            cursor=connection.cursor()
            sql1 = 'select * from mobile'
            cursor.execute(sql1)
            data2 = cursor.fetchall()
            file_like2 = io.BytesIO(data2[0][0])

            img1=Image.open(file_like2)
            return render_template("image.html",data="img1")
        return render_template("image.html")




face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi





def face_identifire():
    size = 4
    haar_file = 'haarcascade_frontalface_default.xml'
    datasets = 'datasets'
  
# Part 1: Create fisherRecognizer 
    print('Recognizing Face Please Be in sufficient Lights...') 
  
# Create a list of images and a list of corresponding names 
    (images, lables, names, id) = ([], [], {}, 0) 
    for (subdirs, dirs, files) in os.walk(datasets): 
        for subdir in dirs: 
            names[id] = subdir 
            subjectpath = os.path.join(datasets, subdir) 
            for filename in os.listdir(subjectpath): 
                path = subjectpath + '/' + filename 
                lable = id
                images.append(cv2.imread(path, 0)) 
                lables.append(int(lable)) 
            id += 1
    (width, height) = (130, 100) 
  
# Create a Numpy array from the two lists above 
    (images, lables) = [np.asarray(lis) for lis in [images, lables]] 
  
# OpenCV trains a model from the images 
# NOTE FOR OpenCV2: remove '.face' 
    model = cv2.face.LBPHFaceRecognizer_create() 
    model.train(images, lables) 
  
# Part 2: Use fisherRecognizer on camera stream 
    face_cascade = cv2.CascadeClassifier(haar_file) 
    webcam = cv2.VideoCapture(0) 
    c=0
    x=0
    id=""
    while True: 
        ret, im = webcam.read() 
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        confidence=0
        for (x, y, w, h) in faces: 
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            face = gray[y:y + h, x:x + w] 
            face_resize = cv2.resize(face, (width, height)) 
        # Try to recognize the face 
            prediction = model.predict(face_resize) 
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3) 
  
            if prediction[1]<500: 
                confidence = int(100*(1-(prediction[1])/300))
                print(confidence)
            
            if confidence>75:
                
                cv2.putText(im, '% s - %.0f' % (names[prediction[0]], prediction[1]), (x-10, y-10),  cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                id=names[prediction[0]]
                x=1
                
                
            else: 
                cv2.putText(im, 'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
                c=c+1
                print("count=",c)
        cv2.imshow('OpenCV', im) 
        key = cv2.waitKey(1)
        if x==1:
            webcam.release()
            cv2.destroyAllWindows()
            return(1,id)
            break
        elif key == 27 or c==100:
            webcam.release()
            cv2.destroyAllWindows()
            return(0,0)
            
            break
        
    
    
    








def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face



def createfacedataset():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT max(id) FROM signup3")
    rs=cursor.fetchone()
    datasets = 'datasets' 
    rs=int(rs['max(id)'])
    rs=rs+1
    sub_data = str(rs)
    cap = cv2.VideoCapture(0)
    path = os.path.join(datasets, sub_data) 
    if not os.path.isdir(path): 
        os.mkdir(path) 
    count = 0
    while True:
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is not None:
            for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
                
                face = cv2.resize(frame[y:y+h, x:x+w],(200,200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('% s/% s.jpg' % (path, count), face) 
                count+=1
                cv2.putText(frame,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                #cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
                
        else:
            print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==100:
            break
        cv2.imshow('identifying',frame)
    
    cv2.destroyWindow('identifying')
    cap.release()
    
    print('Colleting Samples Complete!!!')





# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST' and 'username' in request.form and 'email' in request.form and 'date' in request.form and 'password' in request.form and 'cpassword' in request.form and 'phone' in request.form :
        # Create variables for easy access
        username = request.form['username']
        email = request.form['email']
        date=request.form['date']
        password =sha256_crypt.encrypt(str(request.form['password'])) 
        cpassword = request.form['cpassword']
        mobile = request.form['phone']
        createfacedataset()

                # Check if account exists using MysSQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM signup WHERE email = %s', (email,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        msg = ''
        error=''

        if account:
            error = 'Account already exists!'
            return render_template('register1.html', error=error)
        else:
            # create cursor
            cur = mysql.connection.cursor()

                # Execute query
            cur.execute("INSERT INTO signup1(id,username, email, dates, password,phone) VALUES(%s, %s, %s, %s, %s)", (username, email, date, password,mobile))

                # Commit to DB
            mysql.connection.commit()

                # Close connection
            cur.close()

            flash('You are now registered and can log in', 'success')
            return redirect(url_for('login'))
    return render_template('register1.html')

# User Register

@app.route('/signup2', methods = ['GET', 'POST'])
def voter_register():
    if request.method == 'POST':
        firstname = request.form['first_name']
        lastname = request.form['last_name']
        email=request.form['email']
        password = sha256_crypt.encrypt(str(request.form['password']))
        dob = request.form['birthday']
        gender = request.form['gender']
        phone = request.form['phone']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM signup3 WHERE email=%s",[email])
        rs=cursor.fetchone()
        
        if rs is None:
            createfacedataset()
            cursor.execute("INSERT INTO signup3(firstname,lastname,email,password,dob,gender,phone) VALUES (%s,%s,%s,%s,%s,%s,%s)",(firstname, lastname,email,password,dob,gender,phone))
            
            mysql.connection.commit()

            flash("Successfully registered and face is captuted",'success')
            return redirect('/login')
        else :
            #app.run(debug=True)
            return("User already Exist")
        
    else :
        return render_template('signup2.html')


@app.route('/viewarticle')
def articles():
    # Create cursor
    cur = mysql.connection.cursor()

    # Get articles
    result = cur.execute("SELECT * FROM addarticle")

    articles = cur.fetchall()

    if result > 0:
        return render_template('viewarticle.html', articles=articles)
    else:
        msg = 'No Articles Found'
        return render_template('viewarticle.html', msg=msg)
    # Close connection
    cur.close()

@app.route('/article/<string:id>/')
def article(id):
    # Create cursor
    cur = mysql.connection.cursor()

    # Get article
    result = cur.execute("SELECT * FROM addarticle WHERE id = %s", [id])

    article = cur.fetchone()

    return render_template('article.html', article=article)



@app.route('/face',methods=['GET','POST'])
def fingerPrint():
    if request.method == 'POST':
        a,b=face_identifire()
        session['id']=b
        print(a)
        print(b)
        if a==1:
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT id FROM signup3 WHERE id=%s",[b])
            rs=cursor.fetchone()
            print(a)
            if(rs is not None):
                print(a)
                flash("successfully face reconized",'success')
                return redirect(url_for("dashboard"))
            elif a==0:
                flash("Your face is not recognized",'danger')
                return redirect('/login')
    return render_template('face.html')


@app.route('/login', methods=['GET', 'POST'])

def login():
        if request.method=='POST':
                # Get Form Fields

                email = request.form['email']

                password_candidate = request.form['password']

                # Create cursor

                cur = mysql.connection.cursor()
                # Get user by username

                result = cur.execute("SELECT * FROM signup3 WHERE email = %s", [email])

                msg = ''
                error=''
                if result > 0:
                        # Get stored hash

                        data = cur.fetchone()
                        password = data['password']


                        # Compare Passwords

                        if sha256_crypt.verify(password_candidate, password):
                                session['log'] = True

                                session['email'] = email
                                flash("successfully login the first level of authentication",'success')
                                return redirect('/face')
                        else:
                                error = 'Invalid login'
                                return render_template('login1.html', error=error)
                        cur.close()
                else:
                        error="username not found"
                        return render_template('login1.html',error=error)
        return render_template('login1.html')


@app.route('/adminlogin', methods = ['GET', 'POST'])
def adminlogin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * from adminlogin WHERE email=%s AND password=%s",(username,password))
        data = cursor.fetchone()
        error=''
        if data is not None:
            flash('Successfully Loggedin, Welcome Admin!!!', 'success')
            return redirect('/adminhome')
        else :
            error="Invalid Admin Credentials"
            return render_template('adminlogin.html',e=error)
    else :
        return render_template('adminlogin.html')

@app.route('/adminhome')
def adminhome():
        return render_template("adminhome.html")

@app.route('/userinfo')
def userinfo():
    cursor=mysql.connection.cursor()
    cursor.execute("select * from signup3")
    rows=cursor.fetchall()
    return render_template("userinfo.html",r=rows)

@app.route('/deleteinfo', methods = ['GET', 'POST'])
def deleteinfo():
    if request.method == 'POST':
        email = request.form['email']
        cursor = mysql.connection.cursor()
        result = cursor.execute("DELETE  from signup3 where email = %s", [email])
        mysql.connection.commit()
        if result > 0:
            data = cursor.fetchone()
            flash('Successfully Deleted User Record', 'success')
            return redirect('/userinfo')
    return render_template('deleteinfo.html')

@app.route('/updateinfo', methods = ['GET', 'POST'])
def updateinfo():
    if request.method == 'POST':
        email = request.form['email']
        cursor = mysql.connection.cursor()
        result = cursor.execute("SELECT *  from signup3 where email = %s", [email])
        mysql.connection.commit()
        if result > 0:
            data = cursor.fetchone()
            return render_template('updatedisplay.html',d=data,email=email)
                
    return render_template('updateinfo.html')




@app.route('/edit_article/<string:id>', methods=['GET', 'POST'])
@is_log_in
def edit_article(id):
    # Create cursor
    cur = mysql.connection.cursor()

    # Get article by id
    result = cur.execute("SELECT * FROM articles WHERE id = %s", [id])

    article = cur.fetchone()
    cur.close()
    # Get form
    form = ArticleForm(request.form)

    # Populate article form fields
    form.title.data = article['title']
    form.body.data = article['body']

    if request.method == 'POST' and form.validate():
        title = request.form['title']
        body = request.form['body']

        # Create Cursor
        cur = mysql.connection.cursor()
        app.logger.info(title)
        # Execute
        cur.execute ("UPDATE articles SET title=%s, body=%s WHERE id=%s",(title, body, id))
        # Commit to DB
        mysql.connection.commit()

        #Close connection
        cur.close()

        flash('Article Updated', 'success')

        return redirect(url_for('dashboard'))

    return render_template('edit_article.html', form=form)



class ArticleForm(Form):
    title = StringField('Title', [validators.Length(min=1, max=200)])
    author=StringField('Author', [validators.Length(min=1, max=20)])
    body = TextAreaField('Body', [validators.Length(min=30)])



@app.route('/addarticle', methods=['GET', 'POST'])
def addarticle():
    form = ArticleForm(request.form)
    if request.method == 'POST' and form.validate():
        title = form.title.data
        author=form.author.data
        body = form.body.data


        # Create Cursor
        cur = mysql.connection.cursor()

        # Execute
        cur.execute("INSERT INTO addarticle(title, author, body) VALUES(%s, %s, %s)",(title,author,body))

        # Commit to DB
        mysql.connection.commit()

        #Close connection
        cur.close()

        flash('Article Created', 'success')

        return redirect(url_for('addarticle'))

    return render_template('addarticle.html', form=form)




# check if user logged in:


@app.route('/logout')
#@is_log_in
def logout():
     session.clear()
     flash('You are logged out','success')
     return redirect('/')

@app.route('/mobile',methods=['GET', 'POST'])
#@is_log_in
def mobile():
        if request.method=='POST':
                a1=int(request.form['mobile1'])
                a2=int(request.form['mobile2'])
                a3=float(request.form['mobile3'])
                a4=int(request.form['mobile4'])
                a5=int(request.form['mobile5'])
                a6=int(request.form['mobile6'])
                a7=int(request.form['mobile7'])
                a8=float(request.form['mobile8'])
                a9=int(request.form['mobile9'])
                a10=int(request.form['mobile10'])
                a11=int(request.form['mobile11'])
                a12=int(request.form['mobile12'])
                a13=int(request.form['mobile13'])
                a14=int(request.form['mobile14'])
                a15=int(request.form['mobile15'])
                a16=int(request.form['mobile16'])
                a17=int(request.form['mobile17'])
                a18=int(request.form['mobile18'])
                a19=int(request.form['mobile19'])
                a20=int(request.form['mobile20'])
                mp=str(s.predict([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20]])[:][0])
        return render_template("mobile.html", v=mp)

@app.route('/mobilespec', methods=['GET', 'POST'])
#@is_log_in
def mobilespec():
            cur1=mysql.connection.cursor()
            cur1.execute("SELECT * from product")
            k=cur1.fetchall()
            return render_template('mobilespec.html',products=k)

@app.route('/laptopspec', methods=['GET', 'POST'])
#@is_log_in
def laptopspec():
            cur1=mysql.connection.cursor()
            cur1.execute("SELECT * from laptopspec")
            k=cur1.fetchall()
            return render_template('laptopspec.html',products=k)

@app.route('/tabletspec', methods=['GET', 'POST'])
#@is_log_in
def tabletpec():
            cur1=mysql.connection.cursor()
            cur1.execute("SELECT * from tabletspec")
            k=cur1.fetchall()
            return render_template('tabletspec.html',products=k)

@app.route('/cameraspec', methods=['GET', 'POST'])
#@is_log_in
def cameraspec():
            cur1=mysql.connection.cursor()
            cur1.execute("SELECT * from cameraspec")
            k=cur1.fetchall()
            return render_template('cameraspec.html',products=k)

@app.route('/viewspec', methods=['GET', 'POST'])
#@is_log_in
def viewspec():
            cur1=mysql.connection.cursor()
            cur2=mysql.connection.cursor()
            cur3=mysql.connection.cursor()
            cur4=mysql.connection.cursor()
            cur1.execute("SELECT * from product")
            cur2.execute("SELECT * from laptopspec")
            cur3.execute("SELECT * from tabletspec")
            cur4.execute("SELECT * from cameraspec")
            k=cur1.fetchall()
            k1=cur2.fetchall()
            k2=cur3.fetchall()
            k3=cur4.fetchall()
            return render_template('viewpersonaldevices.html',p1=k,p2=k1,p3=k2,p4=k3)




@app.route('/laptop',methods=['GET', 'POST'])
#@is_log_in
def laptop():
    if request.method=='POST':
        a1=int(request.form['laptop1'])
        a2=int(request.form['laptop2'])
        a3=int(request.form['laptop3'])
        a4=float(request.form['laptop4'])
        a5=int(request.form['laptop5'])
        a6=int(request.form['laptop6'])
        a7=int(request.form['laptop7'])
        a8=int(request.form['laptop8'])
        a9=float(request.form['laptop9'])
        lp=str(s1.predict([[a1,a2,a3,a4,a5,a6,a7,a8,a9]])[:][0])
        return render_template("laptop.html",d=lp)
    return render_template("laptop.html")

@app.route('/tablet',methods=['GET', 'POST'])
#@is_log_in
def tablet():
        if request.method=='POST':
                a1=int(request.form['tablet1'])
                a2=int(request.form['tablet2'])
                a3=float(request.form['tablet3'])
                a4=int(request.form['tablet4'])
                a5=int(request.form['tablet5'])
                a6=int(request.form['tablet6'])
                a7=int(request.form['tablet7'])
                a8=float(request.form['tablet8'])
                a9=int(request.form['tablet9'])
                a10=int(request.form['tablet10'])
                a11=int(request.form['tablet11'])
                a12=int(request.form['tablet12'])
                a13=int(request.form['tablet13'])
                a14=int(request.form['tablet14'])
                a15=int(request.form['tablet15'])
                a16=int(request.form['tablet16'])
                a17=int(request.form['tablet17'])
                a18=int(request.form['tablet18'])
                a19=int(request.form['tablet19'])
                mp=str(s2.predict([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19]])[:][0])
                return render_template("tablet.html", v=mp)
        return render_template("tablet.html")

@app.route('/camera1',methods=['GET', 'POST'])
#@is_log_in
def camera1():
    if request.method=='POST':
        a1=int(request.form['camera1'])
        a2=int(request.form['camera2'])
        a3=int(request.form['camera3'])
        a4=int(request.form['camera4'])
        a5=int(request.form['camera5'])
        a6=int(request.form['camera6'])
        a7=int(request.form['camera7'])
        a8=int(request.form['camera8'])
        a9=int(request.form['camera9'])
        a10=int(request.form['camera10'])
        a11=int(request.form['camera11'])
        a12=int(request.form['camera12'])
        lp=str(s3.predict([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12]])[:][0])

        return render_template("camera1.html",d=lp)
    return render_template("camera1.html")

@app.route('/contact',methods=['GET', 'POST'])
def contact():
        if request.method=='POST':
            name=request.form['name']
            email=request.form['email']
            phone=request.form['phone']
            msg=request.form['msg']
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO contact(name,email,phone,msg) VALUES(%s, %s, %s,%s)",(name,email,phone,msg))
            mysql.connection.commit()
            cur.close()
            flash('Successfully sent to admin', 'success')
        return render_template("contact.html")

@app.route('/viewmsg')

def viewmsg():
    cursor=mysql.connection.cursor()
    cursor.execute("select * from contact")
    rows=cursor.fetchall()
    return render_template("msgview.html",r=rows)


@app.route('/search1',methods=['GET', 'POST'])
#@is_log_in
def search1():
    if request.method=='POST':
        mobname=request.form['mobname']
        cur1=mysql.connection.cursor()
        cur1.execute("SELECT pid,code,image,category,price,discount from product where name=%s",[mobname])
        r=cur1.fetchone()
        return render_template('search.html',r=r['code'],mobname=mobname,k=r['image'],s=r['category'],t=r['discount'])
    return render_template('search.html')


@app.route('/dashboard')
#@is_log_in
def dashboard():
        return render_template("dashboard.html")






class Dashboard(Form):
        title = StringField('Title', [validators.Length(min=5, max=50)])

        body = TextAreaField('Body', [validators.Length(min=30, max=500)])
 


     
if __name__ == "__main__":
     app.secret_key = 'secret123'
     app.run(debug = True)

