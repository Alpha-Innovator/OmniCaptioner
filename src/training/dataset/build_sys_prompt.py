def build_sys_prompt(flag):
    if flag=='detailed':
        system_prompt =SYSTEM_MESSAGE_Detailed
    if flag=='medium':
        system_prompt =SYSTEM_MESSAGE_Medium   
    if flag=='short':
        system_prompt =SYSTEM_MESSAGE_Short   
    if flag=='tag':
        system_prompt =SYSTEM_MESSAGE_Tag   
    if flag=='detailed_CN':
        system_prompt =SYSTEM_MESSAGE_Detailed_CN   
    if flag=='medium_CN':
        system_prompt =SYSTEM_MESSAGE_Medium_CN   
    if flag=='short_CN':
        system_prompt =SYSTEM_MESSAGE_Short_CN 
    if flag=='tag_CN':
        system_prompt =SYSTEM_MESSAGE_Tag_CN


    if flag=='detailed_natural':
        system_prompt =SYSTEM_MESSAGE_Detailed_Natural
    if flag=='medium_natural' :
        system_prompt =SYSTEM_MESSAGE_Medium_Natural   
    if flag=='short_natural' :
        system_prompt =SYSTEM_MESSAGE_Short_Natural   
    
    if flag=='detailed_natural_CN' :
        system_prompt =SYSTEM_MESSAGE_Detailed_CN_Natural   
    if flag=='medium_natural_CN' :
        system_prompt =SYSTEM_MESSAGE_Medium_CN_Natural   
    if flag=='short_natural_CN' :
        system_prompt =SYSTEM_MESSAGE_Short_CN_Natural 
    
    if flag=='chart':
        system_prompt = SYSTEM_MESSAGE_OCR_chart_math #SYSTEM_MESSAGE_OCR_chart_math  (11k+152k)
    if flag=='table':
        system_prompt = SYSTEM_MESSAGE_OCR_table_math #SYSTEM_MESSAGE_OCR_chart_math  (1M+136k+13k)
    if flag=='structured_CN' : 
        system_prompt = SYSTEM_MESSAGE_OCR_chart_math_CN #SYSTEM_MESSAGE_OCR_chart_math 
        
    if flag=='math_equation' :
        system_prompt = SYSTEM_MESSAGE_OCR_equation_math #SYSTEM_MESSAGE_OCR_vrdu_equation  (5M) 
    if flag=='math_geometry'  :
        system_prompt = SYSTEM_MESSAGE_OCR_mathgeo_math #SYSTEM_MESSAGE_OCR_vrdu_equation  (102k)
    if flag=='chemdata' :
        system_prompt =SYSTEM_MESSAGE_chemdata 

    if flag=='OCR' :
        system_prompt =SYSTEM_MESSAGE_OCR_textqa #SYSTEM_MESSAGE_OCR_poster  
    if flag=='poster':
        system_prompt =SYSTEM_MESSAGE_OCR_Image #SYSTEM_MESSAGE_OCR_poster  
    if flag=='poster_CN':
        system_prompt =SYSTEM_MESSAGE_OCR_Image_CN #SYSTEM_MESSAGE_OCR_poster 
  
    if flag=='video':
        system_prompt =SYSTEM_MESSAGE_VIDEO_ALL #SYSTEM_MESSAGE_OCR_poster  
    
    if flag == 'UI':
        system_prompt = SYSTEM_MESSAGE_UI
    if flag == 'UI_CN':
        system_prompt = SYSTEM_MESSAGE_UI_CN

    return system_prompt