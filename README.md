 # the falcon project
Hello World, We are Vivaan Ghosal and Aayush Dugar trying to develop an affordable NDVI system. 


Developing our falcon agricultural idea, 

take the input- OpenCV

picture input, samples 
samples and making photo frames

generate output of the heat map

what to do with the data

sending data to the farmers

steps after data collection process 


FULL FLOW- 
1. Image data extraction (RGB values, near infared lowk depends on if we do the multispectral camera cause that might change alot
2. Normalisation of values (see if you wanna alter RGB between 0 and 1 if its more efficient mathematically or not ill check)
3. Vegetation index calculations
4. - VARI ( Visible atmospherically resistant index) formula is (g-r)/(g-r+b)
   - TGI (triangular greenness index) is (g-0.39*r-0.61*b)
   - EXG (expected greenness) is (2*g-r-b)
6. OPTIONAL - water stress detection I was thinking we can combine the diff indexes I read this smwhere
7. - def detect_water_stress(vari, exg, gli):
    # Normalize all indices to 0-1 range I think this could be the best way my bad for syntax I lowk forgot numpy
    vari_norm = np.clip(vari, 0, 1)
    exg_norm = np.clip(exg, 0, 1)  
    gli_norm = np.clip(gli, 0, 1)
    
    # Calculate health score (weights based on how acc these numbers are this is what gpt said take w a pinch of salt)
    health_score = 0.4 * vari_norm + 0.3 * exg_norm + 0.3 * gli_norm
    
    # Convert to stress probability (higher is obv more stressed)
    stress_probability = 1.0 - health_score
    
    return stress_probability
8. calculate the health
   # Primary health components (weighted combination)
primary_health = (0.3 * clip(VARI, 0, 1) + # 30% vegetation index
    0.2 * clip(ExG, 0, 1) + # 20% green vegetation
    0.2 * (clip(TGI, -0.5, 0.5) + 0.5) + # 20% chlorophyll content  
    0.1 * clip(vNDVI, 0, 1) # 30% overall index as this and vari are lowk the most accurate and this again depends on wether we do multispectral or na
)

#  stress penalties is optional but like if we have predisposed data then we can incorporate for more accuracy or we can just keep this aside and have it as a future imrpovement or sm shit like that 
health_index = primary_health × (1.0 - 0.4 × water_stress - 0.3 × nutrient_deficiency)

9. Rank the health of the area based on health index and then assign it a colour for the gradient map of plant healthh to represent the crops ka view
