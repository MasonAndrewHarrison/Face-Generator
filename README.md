## Installation
1. Clone this repository:\
   `git clone https://github.com/MasonAndrewHarrison/Face-Generator.git`

2. Change Directory:\
   `cd Face-Generator`
      
4. Create virtual environment:\
   `python -m venv venv`
   
5. Activate it:\
   (Linux)`source venv/bin/activate`\
   (Windows CMD)`venv\Scripts\activate.bat`\
   (Windows Power Shell)`venv\Scripts\Activate.ps1`
   
7. Install PyTorch:\
   (For CUDA)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`\
   (For CPU)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
   
9. Install dependencies:\
   `pip install -r requirements.txt`

10. Download Dataset:\
  `python create_dataset.py`
   
12. Run:\
  `python main.py`



## Examples

<table>
  <tr>
    <td><img src="images/img1.png" alt="img1" width="150"/></td>
    <td><img src="images/img2.png" alt="img2" width="150"/></td>
    <td><img src="images/img3.png" alt="img3" width="150"/></td>
    <td><img src="images/img4.png" alt="img4" width="150"/></td>
    <td><img src="images/img5.png" alt="img5" width="150"/></td>
    <td><img src="images/img6.png" alt="img6" width="150"/></td>
  </tr>
</table>
