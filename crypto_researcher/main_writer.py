import os
from groq import Groq
from dotenv import load_dotenv
from news_data_agent import NewsDataAgent, NewsSummaryAgent
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.colors import HexColor
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

class MainWriterAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "mixtral-8x7b-32768"  # A fast, relatively small model
        self.news_agent = NewsDataAgent()
        self.news_summary_agent = NewsSummaryAgent()

    def generate_report(self, news_summary):
        system_prompt = """You are an AI assistant specialized in cryptocurrency research and analysis. Your task is to synthesize information from various sources and generate a comprehensive, well-structured crypto research report. Use the provided summarized news data to create insightful analysis and actionable insights for crypto investors and enthusiasts."""

        user_prompt = f"""Generate a comprehensive crypto research report based on the following summarized news data:

News Summary: {news_summary}

The report should include:
1. Executive Summary
2. Key News and Events
3. Market Implications
4. Potential Opportunities and Risks
5. Conclusion and Outlook

Format the report in markdown with clear headings and subheadings. Ensure the information is concise, relevant, and provides valuable insights for crypto investors and enthusiasts."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )

        return response.choices[0].message.content

    def export_to_pdf(self, report):
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crypto_report_{timestamp}.pdf"

        doc = SimpleDocTemplate(filename, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

        # Update existing styles
        styles['Title'].fontSize = 24
        styles['Title'].spaceAfter = 12
        styles['Title'].textColor = HexColor('#333333')

        styles['Heading1'].fontSize = 18
        styles['Heading1'].spaceAfter = 6
        styles['Heading1'].textColor = HexColor('#444444')

        styles['Heading2'].fontSize = 16
        styles['Heading2'].spaceAfter = 6
        styles['Heading2'].textColor = HexColor('#555555')

        styles['Normal'].fontSize = 12
        styles['Normal'].spaceAfter = 6
        styles['Normal'].textColor = HexColor('#333333')

        content = []

        # Add title
        content.append(Paragraph("Cryptocurrency Market Report", styles['Title']))
        content.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        content.append(Spacer(1, 12))

        # Process the markdown-formatted report
        lines = report.split('\n')
        for line in lines:
            if line.startswith('# '):
                content.append(Paragraph(line[2:], styles['Heading1']))
            elif line.startswith('## '):
                content.append(Paragraph(line[3:], styles['Heading2']))
            elif line.strip() == '':
                content.append(Spacer(1, 6))
            else:
                content.append(Paragraph(line, styles['Normal']))

        doc.build(content)
        print(f"Report exported to {filename}")
        return filename

    def run(self):
        # Fetch news data
        news_data = self.news_agent.run()
        
        # Summarize news data
        news_summary = self.news_summary_agent.run(news_data)
        
        # Generate report
        report = self.generate_report(news_summary)
        
        # Export report to PDF
        pdf_filename = self.export_to_pdf(report)
        
        return report, pdf_filename

if __name__ == "__main__":
    writer = MainWriterAgent()
    final_report, pdf_file = writer.run()
    print(f"Report generated successfully. Check the PDF file: {pdf_file}")
