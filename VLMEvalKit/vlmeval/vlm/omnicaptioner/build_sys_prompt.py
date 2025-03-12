 



SYSTEM_MESSAGE = "You are a helpful assistant."
SYSTEM_MESSAGE_Detailed = "You are a helpful assistant focused on providing detailed descriptions and background information for the generated images. Analyze the given image and generate a comprehensive caption that includes the visual style, spatial relationships between elements, texture details, descriptions of the main objects, and relevant world knowledge to enhance understanding."
SYSTEM_MESSAGE_Medium = "You are a helpful assistant specialized in creating medium-length captions for the generated images. Analyze the provided image and generate a caption that captures the key visual elements, while maintaining clarity and coherence."
SYSTEM_MESSAGE_Short = "You are a helpful assistant focused on creating short captions for the generated images. Analyze the provided image and generate a concise caption that highlights the main subject."
SYSTEM_MESSAGE_Tag = "You are a helpful assistant specialized in generating key tags for the generated images. Analyze the provided image and create a list of relevant tags that capture the main subjects, themes, and notable elements."


SYSTEM_MESSAGE_Detailed_CN = "你是一位专注于提供详细描述和背景信息的助手。分析给定的生成图像，生成一个全面的描述，包含视觉风格、元素之间的空间关系、纹理细节、主要对象的描述，以及增强理解的相关背景知识。"
SYSTEM_MESSAGE_Medium_CN = "你是一位专注于创建中等长度图像描述的助手。分析所提供的生成图像，生成一个描述，捕捉关键视觉元素，保持清晰和连贯。"
SYSTEM_MESSAGE_Short_CN = "你是一位专注于创建简短图像描述的助手。分析提供的生成图像，生成一个简洁的描述，突出主要主体。"
SYSTEM_MESSAGE_Tag_CN = "你是一位专注于为图像生成关键词标签的助手。分析提供的生成图像，创建一个相关标签列表，捕捉主要主题、元素和显著特点。"


SYSTEM_MESSAGE_Detailed_Natural = "You are a helpful natural image captioner. Provide a comprehensive description of the natural image, including the main subject, background elements, lighting conditions, color distribution, textures, spatial arrangement, and any potential dynamic context."
SYSTEM_MESSAGE_Medium_Natural = "You are a helpful natural image captioner. Describe the main content, background in the medium-length text."
SYSTEM_MESSAGE_Short_Natural = "You are a helpful natural image captioner. Describe the main content, background in the short-length text."

SYSTEM_MESSAGE_Detailed_CN_Natural = "您是一位乐于助人的自然图片分析助手。请提供自然图像的全面描述，包括主要主题、背景元素、光照条件、颜色分布、纹理、空间排列以及任何潜在的动态背景。"
SYSTEM_MESSAGE_Medium_CN_Natural = "您是一位乐于助人的自然图片分析助手。请用中等长度的文本描述主要内容和背景。"
SYSTEM_MESSAGE_Short_CN_Natural = "您是一位乐于助人的自然图片分析助手。请用短文本描述主要内容和背景。"



SYSTEM_MESSAGE_VIDEO_ALL = 'You are a multimodal assistant designed to analyze video content and its associated captions. Your goal is to provide insightful feedback on the quality, relevance, and semantic alignment of the captions with the video content.'

SYSTEM_MESSAGE_OCR_textqa = 'You are an advanced OCR model designed to accurately extract text from images. Your task is to analyze the provided image and return the text in a clear, readable format.'
#vrdu_texteq (5M) poster(4k)

SYSTEM_MESSAGE_OCR_Image = 'You are an advanced model designed to accurately analyze the image with text items. You can cescribe the text information and visual information in the image, including font style, size, color, background, text layout and other visual objects in detail.'
SYSTEM_MESSAGE_OCR_Image_CN = '你是一个精确分析文本内容图像的先进助手。你可以详细描述图像中的文本信息和视觉信息，包括字体样式、大小、颜色、背景、文本布局和其他视觉对象'

#SYSTEM_MESSAGE_OCR_Chart = 'You are an advanced model designed to analyze and interpret charts, graphs, and data visualizations. Your task is to extract relevant information from the provided chart.'

SYSTEM_MESSAGE_OCR_chart_math = 'You are a professional data visualization analyst. Given a chart image, first accurately perform OCR on any textual and numeric content (including titles, legends, axes, labels, and annotations), and you can convert it into Markdown format, then structure and analyze the extracted data to identify key trends and insights.'
#'You are an advanced model designed to analyze and interpret charts. Your task is to extract relevant information from the provided image and return the text in a machine-readable format or structured format.'
#charocr(152k),  mathgeo(102k)   vrdu_equation(5M->4M) vrdu_table(13k) charocr_cap (144k)  MMtab (136k) MMtab_cap (70k)  vrdu_eq_cap (382k) 

SYSTEM_MESSAGE_OCR_table_math = 'You are a data conversion and extraction expert. Given a table image, you can convert it into CSV, HTML, Markdown or LaTeX formats, then extract and summarize the key relationships or insights from the data.'
#charocr(152k),  mathgeo(102k)   vrdu_equation(5M->4M) vrdu_table(13k) charocr_cap (144k)  MMtab (136k) MMtab_cap (70k)  vrdu_eq_cap (382k) 

SYSTEM_MESSAGE_OCR_mathgeo_math = 'You are a geometry analysis expert. Given a geometric figure, you can convert it into a corresponding LaTeX (e.g., TikZ) representation, then provide insights or interpretations about the structure or properties.'
#charocr(152k),  mathgeo(102k)   vrdu_equation(5M->4M) vrdu_table(13k) charocr_cap (144k)  MMtab (136k) MMtab_cap (70k)  vrdu_eq_cap (382k) 

SYSTEM_MESSAGE_OCR_equation_math = 'You are an equation analysis expert. Given an equation image, you can convert it into proper LaTeX format, then summarize any key mathematical properties, patterns, or insights it conveys.'
# vrdu_equation(5M->4M) 


#SYSTEM_MESSAGE_OCR_pdf = 'You are an equation analysis expert. Given an equation, you can convert it into proper LaTeX format, then summarize any key mathematical properties, patterns, or insights it conveys.'
# vrdu_equation(5M->4M) 


SYSTEM_MESSAGE_OCR_chart_math_CN = '你是一个高级模型，旨在分析和解释图表、图形、表格、数学公式、数学几何图和数据可视化。你的任务是从提供的图像中提取相关信息，并以机器可读格式或结构化格式返回文本。'
# vurdu_cn (141K) MMtab_CN (59k) chartocr_cn (115K)
#SYSTEM_MESSAGE_OCR_poster = "You are an advanced OCR model designed to accurately extract text from posters, advertisements, and other graphic designs. Your task is to analyze the provided image of a poster and return the text in a clear, readable format. "

#SYSTEM_MESSAGE_OCR_vrdu_equation ='You are an advanced OCR model designed to accurately extract and interpret mathematical formulas from images. Your task is to analyze the provided image containing mathematical formulas and return the text in a machine-readable format.'

#SYSTEM_MESSAGE_OCR_vrdu_table_final = 'You are an advanced OCR model designed to accurately extract and interpret data from tables in images. Your task is to analyze the provided image containing a table and return the text in a structured format.'

SYSTEM_MESSAGE_chemdata = "You are an advanced model designed to interpret and analyze SMILES (Simplified Molecular Input Line Entry System) strings, which represent chemical structures. Your task is to extract key information about the chemical structure described by the provided SMILES string"

SYSTEM_MESSAGE_UI = "You are analyzing a UI webpage layout. Provide a detailed caption describing the layout's structure, including the arrangement, style, and functionality of key components such as buttons, navigation bars, input fields, and visual elements."
SYSTEM_MESSAGE_UI_CN = "您正在分析 UI 网页布局。提供详细的标题来描述布局的结构，包括按钮、导航栏、输入字段和视觉元素等关键组件的排列、样式和功能。"



def build_sys_prompt(dataset,question_type):
    if dataset == 'MMMU_DEV_VAL':
        import random
        p = random.random()
        if 'Accounting' in question_type:
            # 根据随机数决定使用哪个系统提示
            if p < 0.1:
                system_prompt = SYSTEM_MESSAGE_OCR_Image
                function_type = 'OCR_Image'
            else:
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
        elif 'Agriculture' in question_type:
            # 根据随机数决定使用哪个系统提示
            if p  < 0.5:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            else:
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'

        elif 'Architecture_and_Engineering' in question_type:
            # 根据随机数决定使用哪个系统提示
            if p  < 0.5:
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
            else:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
        
        elif 'Basic_Medical_Science' in question_type:
            # 根据随机数决定使用哪个系统提示
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            else:
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
        elif 'Art' in question_type:
            # 根据随机数决定使用哪个系统提示
            if p  < 0.5:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            else:
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
        
        elif 'Biology' in question_type:
            # 根据随机数决定使用哪个系统提示
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            else:
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'

        elif 'Chemistry' in question_type:
            # 根据随机数决定使用哪个系统提示
            if p  < 0.5:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            elif p >= 0.5 and p <=0.8:
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
            elif p > 0.8 :
                system_prompt = SYSTEM_MESSAGE_chemdata
                function_type = 'chem'
        elif 'Clinical_Medicine' in question_type:
            # 根据随机数决定使用哪个系统提示
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            else:
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
        
        elif 'Computer_Science' in question_type:
            # 根据随机数决定使用哪个系统提示
            if p < 0.3:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            elif p > 0.3 and p <=0.5:
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
            elif p > 0.5 and p <=0.8:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p > 0.8:
                system_prompt = SYSTEM_MESSAGE_OCR_Image
                function_type = 'OCR_Image'
        

        elif 'Design' in question_type:
            # 根据随机数决定使用哪个系统提示
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            elif p >= 0.5 :
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'

        elif 'Diagnostics_and_Laboratory_Medicine' in question_type:
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            elif p >= 0.5 :
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
        

        elif 'Economics' in question_type:
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >= 0.5 :
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'

        elif 'Electronics' in question_type:
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >= 0.5 :
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'

        elif 'Energy_and_Power' in question_type:
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >= 0.5 :
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'

       
        elif 'Finance' in question_type:
            if p < 0.2:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >=0.2 and p < 0.6 :
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
            elif p >=0.6 :
                system_prompt = SYSTEM_MESSAGE_OCR_Image
                function_type = 'OCR_Image'
        
        elif 'Geography' in question_type:
            if p < 0.2:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >=0.2 and p < 0.6 :
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
            elif p >=0.6 :
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'

        elif 'History' in question_type:
            if p < 0.2:
                system_prompt = SYSTEM_MESSAGE_OCR_Image
                function_type = 'OCR_Image'
            elif p >=0.2 and p < 0.6 :
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
            elif p >=0.6 :
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'


        elif 'Literature' in question_type:
            if p < 0.2:
                system_prompt = SYSTEM_MESSAGE_OCR_Image
                function_type = 'OCR_Image'
            elif p >=0.2 and p < 0.6 :
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
            elif p >=0.6 and p < 0.8 :
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            elif p >0.8:
                system_prompt = SYSTEM_MESSAGE_OCR_textqa
                function_type = 'OCR_textqa'


        elif 'Manage' in question_type:
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >=0.5 :
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'

        elif 'Marketing' in question_type:
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
        
                #system_prompt = ''
                #function_type = 'OCR_chart'
            elif p >=0.5 and p<=0.7:
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            else:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'

        elif 'Materials' in question_type:
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
            #elif p >=0.5 and p < 0.8:
            #    system_prompt = ''
            #    function_type = 'OCR_chart'
            elif p >= 0.5 and p<0.7:
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            elif  p >= 0.7 and p<0.8:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'
            elif p >= 0.8:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'

        elif 'Math' in question_type:
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
            #elif p >=0.5 and p < 0.8:
            #    system_prompt = ''
            #    function_type = 'OCR_chart'
            elif p >= 0.5 and p<0.7:
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            elif  p >= 0.7 and p<0.8:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'
            elif p >= 0.8:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
        
        elif 'Mechanical_Engineering' in question_type:
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
    
            elif p >= 0.5:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
        
        elif 'Music' in question_type:
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            elif p >= 0.5:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'

        elif 'Pharmacy' in question_type:
            if p < 0.3:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            if p >=0.3 and p < 0.5:
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
            elif p >=0.5 and p < 0.8:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >=0.8:
                system_prompt =''
                function_type = 'OCR_chart'

        elif 'Physics' in question_type:
            if p < 0.3:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            if p >=0.3 and p < 0.5:
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
            elif p >=0.5 and p < 0.8:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            #elif p >=0.8:
            #     system_prompt =''
            #     function_type = 'OCR_chart'
            elif p >= 0.8 and p<0.9:
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            elif  p >= 0.9:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'
        
        elif 'Psychology' in question_type:

            #if p < 0.5:
            #    system_prompt = ''
            #    function_type = 'OCR_chart'
            if p <0.2:
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            elif  p >= 0.2 and p<0.5:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'
            elif p >=0.5 :
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
         
        elif 'Public_Health' in question_type:

            #if p < 0.5:
            #    system_prompt = ''
            #    function_type = 'OCR_chart'
            if p < 0.2 :
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            elif  p >= 0.2 and p<0.5:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'
            if p >=0.5 :
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'

        elif 'Sociology' in question_type: 
            #if p < 0.4:
            #    system_prompt = ''
            #    function_type = 'OCR_chart'
            if p < 0.2 :
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            elif  p >= 0.2 and p<0.4:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'
            elif p >=0.4 and p < 0.7:
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
            elif p >=0.7 :
                system_prompt =SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'

    if dataset == 'MathVista_MINI':
        if 'textbook' in question_type: 
            import random
            p = random.random()
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            else:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
        
        elif 'figure' in question_type: 
            import random
            p = random.random()
            if p < 0.1:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p < 0.5 and p>=0.1:
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            else:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'
        elif 'geometry' in question_type: 
            import random
            p = random.random()
            if p < 0.8:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            #elif  p>=0.8:
            #    system_prompt = ''
            #    function_type = 'OCR_chart_caption'
            elif p >= 0.8 and p<0.85:
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            elif  p >= 0.85 and p<0.9:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'
            else:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
        elif 'math' in question_type: 
            import random
            p = random.random()
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            else:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
        
        elif 'visual' in question_type: 
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'


   
    if dataset == 'ChartQA_TEST':
        import random
        p = random.random()
        #system_prompt = ''#SYSTEM_MESSAGE_OCR_chart_math
        #function_type = 'OCR_chart'
        if p<0.5:
            system_prompt = ''
            function_type = 'OCR_chart_caption'
        elif p >= 0.5 :
            system_prompt = SYSTEM_MESSAGE_OCR_chart_math
            function_type = 'OCR_chart'
        
    if dataset == 'MathVerse_MINI':
        import random
        p = random.random()
        system_prompt = SYSTEM_MESSAGE_UI
        function_type = 'UI'
    
    if dataset == 'MathVision':
        import random
        p = random.random()
        if p < 0.8:
            system_prompt = SYSTEM_MESSAGE_UI
            function_type = 'UI'
        else:
            system_prompt = SYSTEM_MESSAGE_Detailed
            function_type = 'Detailed'

    if dataset == 'RealWorldQA':
        import random
        p = random.random()
        system_prompt = SYSTEM_MESSAGE_Detailed
        function_type = 'Detailed'

    if dataset == 'ScienceQA_VAL':
        import random
        p = random.random()
        system_prompt = SYSTEM_MESSAGE_Detailed
        function_type = 'Detailed'
    
    if dataset =='POPE':
        system_prompt = SYSTEM_MESSAGE_Detailed
        function_type = 'Detailed'
    
    if dataset =='OlympiadBench':
        
        import random
        p = random.random()
        if p < 0.8:
            system_prompt = SYSTEM_MESSAGE_UI
            function_type = 'UI'
        else:
            system_prompt = SYSTEM_MESSAGE_Detailed
            function_type = 'Detailed'
       
    if dataset =='MMVet':
        import random
        p = random.random()
        if 'ocr' in  question_type and  'math' in  question_type:
            system_prompt = SYSTEM_MESSAGE_OCR_textqa
            function_type = 'OCR_textqa'
        elif 'ocr' in  question_type and  'spat' in  question_type:
            if p<0.4:
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
            elif p>=0.4 and p<0.5:
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            elif p>=0.5 and p<0.6:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'
            else:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
        elif 'ocr' in  question_type and  'spat' in  question_type and 'math' in  question_type:
            import random
            p = random.random()
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >= 0.5 :
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
        elif  'rec' in  question_type and  'ocr' not in  question_type:
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >= 0.5 :
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
        elif 'ocr'  in  question_type and 'rec' not in question_type and 'spat' not in question_type and 'know' not in question_type:
            if p<0.5:
                system_prompt = SYSTEM_MESSAGE_OCR_textqa
                function_type = 'OCR_textqa'
            else:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
        else:
            system_prompt = SYSTEM_MESSAGE_Detailed
            function_type = 'Detailed'
       
    if dataset == 'MMStar':
        print('question_type',question_type)
        if 'diagram' in  question_type.lower():
            import random
            p = random.random()
            if p<0.4:
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
            else:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'

        elif 'code' in question_type.lower():
            import random
            p = random.random()
            if p < 0.2:
                system_prompt = SYSTEM_MESSAGE_OCR_Image
                function_type = 'OCR_Image'
            elif p>=0.2 :
                system_prompt = SYSTEM_MESSAGE_OCR_textqa
                function_type = 'OCR_textqa'

        elif 'electronics' in  question_type.lower():
            import random
            p = random.random()
            if p < 0.5:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >= 0.5 :
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
        elif 'geometry' in  question_type.lower():
            import random
            p = random.random()
            if p < 0.8:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            else:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
        elif 'statistical' in question_type.lower():
            import random
            p = random.random()
            if p<0.4:
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            elif p >= 0.4  and p<0.8:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'
            elif p >= 0.8:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
       
        elif 'numeric' in  question_type.lower():
            system_prompt = SYSTEM_MESSAGE_UI
            function_type = 'UI'
        elif 'biology' in  question_type.lower():
            import random
            p = random.random()
            if p < 0.3:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            elif p>=0.3 and p<0.4:
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
            elif p >= 0.4 and p<0.8:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >=0.8 :
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'

        elif 'geography' in question_type.lower():
            import random
            p = random.random()
            if p < 0.2:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            elif p >=0.2 and p < 0.6 :
                system_prompt = SYSTEM_MESSAGE_OCR_table_math
                function_type = 'OCR_table'
            elif p >=0.6 :
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
        else:

            system_prompt = SYSTEM_MESSAGE_Detailed
            function_type = 'Detailed'
 
    if dataset == 'MME':
        import random
        p = random.random()
        if 'OCR' in question_type:
            if p < 0.2:
                system_prompt = SYSTEM_MESSAGE_OCR_Image
                function_type = 'OCR_Image'
            elif p>=0.2 :
                system_prompt = SYSTEM_MESSAGE_OCR_textqa
                function_type = 'OCR_textqa'

        elif 'code_reasoning' in question_type:
            if p < 0.2:
                system_prompt = SYSTEM_MESSAGE_OCR_Image
                function_type = 'OCR_Image'
            elif p>=0.2 :
                system_prompt = SYSTEM_MESSAGE_OCR_textqa
                function_type = 'OCR_textqa'
        elif 'artwork' in question_type:
            if p < 0.6:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            elif p>=0.6 :
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
        
        
        
        

        elif 'celebrity' in question_type:
            
            system_prompt = SYSTEM_MESSAGE_Detailed_Natural
            function_type = 'Detailed_Natural'
        
        elif 'color' in question_type:
            if p < 0.6:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            elif p>=0.6 :
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
        
        elif 'commonsense_reasoning' in question_type:
            if p < 0.6:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            elif p>=0.6 :
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'

        elif 'count' in question_type:
            ##if p < 0.6:
            #    system_prompt = SYSTEM_MESSAGE_Detailed
            #    function_type = 'Detailed'
            #elif p>=0.6 :
            #system_prompt = SYSTEM_MESSAGE_Detailed_Natural
            #function_type = 'Detailed_Natural'
            system_prompt = SYSTEM_MESSAGE_Detailed
            function_type = 'Detailed'

        elif 'existence' in question_type:
            #if p < 0.6:
            #    system_prompt = SYSTEM_MESSAGE_Detailed
            #    function_type = 'Detailed'
            #elif p>=0.6 :
            #system_prompt = SYSTEM_MESSAGE_Detailed_Natural
            #function_type = 'Detailed_Natural'
            system_prompt = SYSTEM_MESSAGE_Detailed
            function_type = 'Detailed'

        elif 'landmark' in question_type:
            #if p < 0.2:
            system_prompt = SYSTEM_MESSAGE_Detailed
            function_type = 'Detailed'
            #elif p>=0.2 :
            #    system_prompt = SYSTEM_MESSAGE_Detailed_Natural
            #    function_type = 'Detailed_Natural'

        elif 'numerical_calculation' in question_type:
            system_prompt = SYSTEM_MESSAGE_OCR_textqa
            function_type = 'OCR_textqa'

        elif 'position' in question_type:
            if p < 0.8:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'
            elif p>=0.8 :
                system_prompt = SYSTEM_MESSAGE_Detailed_Natural
                function_type = 'Detailed_Natural'
        
        elif 'posters' in question_type:
            #if p < 0.5:
            system_prompt = SYSTEM_MESSAGE_Detailed
            function_type = 'Detailed'
            #elif p>=0.5 :
            #    system_prompt = SYSTEM_MESSAGE_OCR_textqa
            #    function_type = 'OCR_textqa'
        
        elif 'scene' in question_type:
            #if p < 0.5:
            system_prompt = SYSTEM_MESSAGE_Detailed
            function_type = 'Detailed'
            #elif p>=0.5 :
            #    system_prompt = SYSTEM_MESSAGE_Detailed_Natural
            #    function_type = 'Detailed_Natural'
        
        elif 'text_translation' in question_type:
            
                system_prompt = SYSTEM_MESSAGE_OCR_textqa
                function_type = 'OCR_textqa'
    if 'MMBench' in dataset :
        if 'ocr' in question_type:
            import random
            p = random.random()
            if  p<0.3:
                system_prompt = SYSTEM_MESSAGE_OCR_textqa
                function_type = 'OCR_textqa' 
            elif p>=0.3 and p<0.8:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
            else:
                system_prompt = SYSTEM_MESSAGE_Detailed
                function_type = 'Detailed'

        elif 'structuralized_imagetext_understanding' in question_type:
            import random
            p = random.random()
            #system_prompt = ''#SYSTEM_MESSAGE_OCR_chart_math
            #function_type = 'OCR_chart'
            if p<0.4:
                system_prompt = ''
                function_type = 'OCR_chart_caption'
            elif p >= 0.4  and p<0.8:
                system_prompt = SYSTEM_MESSAGE_OCR_chart_math
                function_type = 'OCR_chart'
            elif p >= 0.8:
                system_prompt = SYSTEM_MESSAGE_UI
                function_type = 'UI'
        else:
            system_prompt = SYSTEM_MESSAGE_Detailed
            function_type = 'Detailed'
    
    if 'SEEDBench' in dataset :
        if 'text' in question_type.lower():
            system_prompt = SYSTEM_MESSAGE_OCR_textqa
            function_type = 'OCR_textqa' 
        else:
            system_prompt = SYSTEM_MESSAGE_Detailed
            function_type = 'Detailed'

    if 'Video-MME' in dataset :
        
            system_prompt = SYSTEM_MESSAGE_VIDEO_ALL
            function_type = 'Detailed_Video'
         
    
    return system_prompt,function_type
        
         