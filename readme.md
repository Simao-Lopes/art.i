![logo](https://raw.githubusercontent.com/Simao-Lopes/art.i/main/Images/New%20banner.png)

# Your ai art buddy

Not everyone has to be an art connaisseur, so with art.i you provide the painting and I tell you the movement of your painting and I give 3 suggestions based on the colours of your painting because you probably really like them.
If you are curious about colour, check our plots folder and see the PowerPoint presentation; you'll be amazed at how colour taste changed over time.

## Usage  
   
The main component is the file **06 - Recommender.** Just run the script insert your image path or link, and just see the magic happen.  
3 paintings will be recommended as well as the colour palette of the image provided, to do this the script loads a pre-made neural network model and classifies the image provided, loads another model to classify again the painting now referring to the colour palette so the recommendation isn't random, the chosen paintings will be from the same movement and will share the same colour palette.

## Developer instructions

### Files included
   
- **0 - Obtaining data.** This notebook contains all the web scraping to get all the artists by movement from the web and the usage of the Chicago museum to build a dataset of images to recommend in the end.
- **1 - Identifying the classes and final image EDA.** As the name suggests in this notebook I identify the top classes to include in the classification model and fill in some missing values, also checking for invalid files.
- **2 - Model.** Building the first model as a baseline using fastai, a fantastic library that makes the process of building neural networks really simple. I will use this just as a baseline of performance and to learn some parameters to use further ahead.
- **2_1 - Resnet.** Modeling of the final model, tested 2 CNN architectures: VGG19 and RESnet50. As the name implies, the best was RESnet50, so I used it ahead. All processes explained in detail inside the notebook
- **3 - Building color palettes.** In this notebook I create color palettes for all images in our datasets. Used knn to cluster all colours in the image array and then picked up the centroid colours.
- **4 - Palette Classifying.** In the previous step I calculated all color palettes, here I'm making clusters of those palettes, so we are making groups of similar palettes.
- **5 - Deep color analysis.** I do a lot of plots here and I calculate the most influential color for each movement. The process is described in detail inside the notebook.
- **6 - Recomender.** Just run it, change your desired file to classify in the code and let the magic happen!
- **Recomender.py.** Using a Streamlit instance you can run this script and run the project as a nice web-based app that has a lot more info and detail than the jupyter notebook. Run first a terminal with the code
''' 
streamlit run Recomender.py
'''

### Folders and files

- **CSV's.** All the CSV’s used on the project. 
- **Images.** Folder with various images on small parts of the project and the images of the Chicago museum to use in the recommender. Due to size restrictions, the main recommended image dataset can't be included. If you need it, send me a message and I’ll gladly send it.
- **Movement Plots.** Repository for all the plots I did, check them out to see the evolution of colour usage through time.
- **Presentation.pptx** Powerpoint containing my conclusions regarding colour usage through time. Also some technical insight and some funny jokes.

## Used technologies

- Chicago Museum API
- Web Scraping (bs4+json)
- Python
- Streamlit
- Keras
- Tensor Flow
- Matplotlib/Seaborn
- PIL

## License

This is an educational project, all materials can be used freely. Reference to the project is welcome
