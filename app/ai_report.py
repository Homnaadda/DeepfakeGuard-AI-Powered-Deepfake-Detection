import google.generativeai as genai
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.units import inch
import os
import datetime
from . import config

# Configure Gemini
genai.configure(api_key=config.GEMINI_API_KEY)

def generate_ai_analysis(video_filename, prediction, confidence, audio_prediction, audio_confidence):
    """
    Generates a textual analysis using Gemini API.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are a senior digital forensics examiner. Generate a professional forensic analysis report for a media file named '{video_filename}'.
        
        **Detection System Results:**
        - **Visual/Video Analysis:** {prediction} (Confidence: {confidence}%)
        - **Audio Analysis:** {audio_prediction} (Confidence: {audio_confidence}%)
        
        Provide the report in the following structured format (plain text, no markdown **bolding** inside the paragraphs, use clear headings):
        
        Section 1: Executive Summary
        [Provide a high-level summary of whether the media is authentic or manipulated.]
        
        Section 2: Technical Assessment
        [Explain the technical implications. If FAKE, mention potential use of GANs, autoencoders, or TTS synthesis. If REAL, mention the consistency of biological signals.]
        
        Section 3: Threat & Risk Implications
        [Discuss the potential impact of this media being circulated.]
        
        Section 4: Verification Recommendations
        [Suggest 2-3 steps for further manual verification.]
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Executive Summary\nAI Analysis unavailable due to error: {str(e)}"

def create_pdf_report(video_filename, prediction, confidence, audio_prediction, audio_confidence, ai_analysis_text, image_paths, output_filename):
    """
    Creates a professional PDF forensic report.
    """
    try:
        doc = SimpleDocTemplate(
            output_filename, 
            pagesize=A4,
            rightMargin=50, leftMargin=50,
            topMargin=50, bottomMargin=50
        )
        
        # Professional Color Palette
        COLOR_PRIMARY = colors.HexColor("#1e3a8a") # Dark Blue
        COLOR_SECONDARY = colors.HexColor("#64748b") # Slate Gray
        COLOR_ACCENT = colors.HexColor("#0ea5e9") # Sky Blue
        COLOR_SUCCESS = colors.HexColor("#166534") # Green
        COLOR_DANGER = colors.HexColor("#991b1b") # Red
        COLOR_LIGHT_BG = colors.HexColor("#f1f5f9") # Light Gray BG
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom Styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=COLOR_PRIMARY,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName="Helvetica-Bold"
        )
        
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=COLOR_SECONDARY,
            alignment=TA_CENTER,
            spaceAfter=30
        )
        
        heading_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=COLOR_PRIMARY,
            spaceBefore=15,
            spaceAfter=10,
            fontName="Helvetica-Bold",
            borderPadding=5,
            borderColor=COLOR_LIGHT_BG,
            borderWidth=0,
            backColor=None 
        )
        
        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
            textColor=colors.black
        )

        # 1. Header Section
        story.append(Paragraph("DIGITAL FORENSIC REPORT", title_style))
        story.append(Paragraph("Deepfake Detection & Media Authentication Protocol", subtitle_style))
        story.append(Spacer(1, 10))
        
        # 2. Case Information Table
        story.append(Paragraph("CASE INFORMATION", heading_style))
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        data_info = [
            ['Case Reference:', f'DF-{abs(hash(video_filename)) % 10000:04d}'],
            ['File Name:', video_filename],
            ['Analysis Date:', timestamp],
            ['Examiner System:', 'EduPath Deepfake AI v1.0']
        ]
        
        t_info = Table(data_info, colWidths=[2.5*inch, 4*inch])
        t_info.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), COLOR_LIGHT_BG),
            ('TEXTCOLOR', (0,0), (0,-1), COLOR_PRIMARY),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ]))
        story.append(t_info)
        story.append(Spacer(1, 20))

        # 3. Detection Results Table
        story.append(Paragraph("FORENSIC EXAMINATION RESULTS", heading_style))
        
        # Determine status icons/text
        vid_status = "⚠️ FAKE / MANIPULATED" if prediction == "FAKE" else "✅ AUTHENTIC / REAL"
        vid_color = COLOR_DANGER if prediction == "FAKE" else COLOR_SUCCESS
        
        aud_status = "⚠️ FAKE / SYNTHETIC" if audio_prediction == "FAKE" else "✅ AUTHENTIC / NATURAL"
        if audio_prediction in ["N/A", "Error"]:
            aud_status = "⚪ NOT ANALYZED"
            aud_color = colors.gray
        else:
            aud_color = COLOR_DANGER if audio_prediction == "FAKE" else COLOR_SUCCESS

        data_results = [
            ['Component', 'Classification', 'Confidence Score', 'Risk Level'],
            ['Video / Visual', vid_status, f"{confidence}%", 'CRITICAL' if prediction == 'FAKE' else 'LOW'],
            ['Audio / Voice', aud_status, f"{audio_confidence}%", 'CRITICAL' if audio_prediction == 'FAKE' else 'LOW']
        ]
        
        t_results = Table(data_results, colWidths=[1.5*inch, 2.5*inch, 1.5*inch, 1*inch])
        t_results.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 11),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('TOPPADDING', (0,0), (-1,0), 12),
            
            # Content Rows
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 10),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, COLOR_LIGHT_BG]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            
            # Specific Colors for Status
            ('TEXTCOLOR', (1,1), (1,1), vid_color),
            ('FONTNAME', (1,1), (1,1), 'Helvetica-Bold'),
            
            ('TEXTCOLOR', (1,2), (1,2), aud_color),
            ('FONTNAME', (1,2), (1,2), 'Helvetica-Bold'),
        ]))
        story.append(t_results)
        story.append(Spacer(1, 25))

        # 4. AI Analysis
        story.append(Paragraph("AI-GENERATED ASSESSMENT", heading_style))
        
        # Simple parsing of the structured text
        sections = ai_analysis_text.split('Section')
        for section in sections:
            if not section.strip(): continue
            
            # Remove the "X:" part if present and get the rest
            parts = section.split(':', 1)
            if len(parts) > 1:
                header_text = parts[0].strip().replace(str(sections.index(section)), '').strip() # Clean up "Section 1" etc if needed
                content_text = parts[1].strip()
                
                # Make header sub-heading
                # Extract the actual title if it looks like "1: Executive Summary"
                # Actually, our prompt says "Section X: Title". split gives us Title.
                
                # Let's just assume the prompt works nicely.
                # parts[0] is " 1" or similar number, we can ignore or format.
                # Actually let's just treat standard lines.
                
                story.append(Paragraph(f"<b>• {parts[0].strip()}</b>" if len(parts[0]) > 2 else f"<b>Analysis Point</b>", body_style))
                story.append(Paragraph(content_text, body_style))
            else:
                story.append(Paragraph(section.strip(), body_style))
                
            story.append(Spacer(1, 5))
            
        story.append(Spacer(1, 20))

        # 5. Evidence
        story.append(Paragraph("EXTRACTED EVIDENCE FRAMES", heading_style))
        story.append(Paragraph("The following key frames were analyzed for artifacts:", body_style))
        story.append(Spacer(1, 10))

        # Images Grid
        images_row = []
        for img_path in image_paths[:4]: # Max 4 images
            if os.path.exists(img_path):
                img = ReportLabImage(img_path, width=3*inch, height=2*inch)
                images_row.append(img)
        
        # Chunk into rows of 2
        img_data = []
        for i in range(0, len(images_row), 2):
            img_data.append(images_row[i:i+2])
            
        if img_data:
            t_images = Table(img_data, colWidths=[3.2*inch, 3.2*inch])
            t_images.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ]))
            story.append(t_images)

        # Footer
        story.append(Spacer(1, 40))
        story.append(Paragraph("End of Report", subtitle_style))
        story.append(Paragraph(f"Generated by EduPath Deepfake Detector | {timestamp}", ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))

        doc.build(story)
        return True
    
    except Exception as e:
        print(f"Error creating PDF: {e}")
        import traceback
        traceback.print_exc()
        return False
