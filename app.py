from flask import Flask,render_template,request,send_from_directory

from model import process_single_image

app=Flask(__name__)

@app.route('/')
def home():
    return render_template("show.html")

IMAGE_DIRECTORY = 'images/'


@app.route('/upload',methods=['POST'])
def upload():
    if request.method=='POST':
        print("post")
    image_file=True
    if 'image' in request.files:
        image_file = request.files['image']
        if image_file.filename != '':
            # Save the file to a specific location on the server
            image_file.save('images/image.jpg')

            model_path = 'RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
            test_img_path = 'images\image.jpg'
            output_path = 'images\image2.jpg'

            process_single_image(model_path, test_img_path, output_path)

            
    print(image_file)
    return render_template("show.html",image=image_file)


@app.route('/input_image')
def get_input_image():
    # Return the image file from the specified directory
    return send_from_directory(IMAGE_DIRECTORY, 'image.jpg')


@app.route('/output_image')
def get_output_image():
    # Return the image file from the specified directory

    return send_from_directory(IMAGE_DIRECTORY, 'image2.jpg')


if(__name__=='__main__'):
    app.run(debug=True,port=8001)
