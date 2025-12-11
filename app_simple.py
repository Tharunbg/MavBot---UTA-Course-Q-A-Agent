"""
UTA Course Q&A Agent - Simplified Gradio Interface
Run with: python app_simple.py
"""

import gradio as gr
import os

# Import your classes
from main_script import CourseQAAgent, AppConfig, DataConfig, ModelConfig


# ============================================================
#  CLASS: Assistant Wrapper
# ============================================================
class GradioCourseAssistant:
    def __init__(self):
        self.agent = None
        self.initialized = False

    def initialize_agent(self):
        """Initialize the course QA agent"""
        try:
            print("🔧 Starting agent initialization...")
            data_config = DataConfig(
                data_file="project_data.csv",
                index_prefix="uta_production",
                chunk_sizes={'courses': 3, 'professors': 3, 'sections': 3}
            )
            print("✓ Data config created")

            config = AppConfig(
                data=data_config,
                log_level="INFO",
                cache_size=1000
            )
            print("✓ App config created")

            print("🤖 Initializing CourseQAAgent (this may take a few minutes)...")
            self.agent = CourseQAAgent(config)
            print("✓ Agent object created")
            
            print("📥 Loading models and building indices...")
            self.agent.initialize()
            print("✓ Agent initialized successfully")
            
            self.initialized = True

            return "✅ Agent initialized! You can now ask questions."

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Initialization error:\n{error_details}")
            return f"❌ Failed to initialize agent:\n{str(e)}\n\nCheck terminal for full error details."

    def process_query(self, query):
        """Process query"""
        if not self.initialized or self.agent is None:
            return "⚠️ Please initialize the agent first by clicking the 'Initialize Agent' button."

        if not query or not query.strip():
            return "⚠️ Please enter a question."

        try:
            response = self.agent.process_query(query)
            return response

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Query error:\n{error_details}")
            return f"❌ Error processing query: {str(e)}"


# ============================================================
#  BUILD GRADIO UI (Simplified)
# ============================================================
def create_gradio_interface():
    assistant = GradioCourseAssistant()

    with gr.Blocks(title="UTA Course Q&A Agent") as demo:

        # Header
        gr.Markdown("""
        <div style="text-align:center;">
            <h1 style="color:#667eea;">🎓 UTA Course Q&A Agent</h1>
            <h3>Course Insights • Professor Analytics • GPA Trends</h3>
        </div>
        """)

        with gr.Row():

            # ---------------- LEFT PANEL ----------------
            with gr.Column(scale=1):
                gr.Markdown("### 🚀 Initialize Assistant")

                init_btn = gr.Button("Initialize Agent", variant="primary")
                init_status = gr.Textbox(
                    label="Status",
                    placeholder="Click Initialize Agent...",
                    interactive=False,
                    lines=3
                )

                gr.Markdown("### 💡 Example Queries")
                gr.Markdown("""
                - CSE 5334
                - Compare CSE 5334 and CSE 5330
                - Easy CS electives
                - Professor John Smith
                - Grade distribution for CSE 5334 Spring 2023
                - Which course is easier, 5334 or 5311?
                - History of CSE 5334
                - Courses similar to Machine Learning
                """)

            # ---------------- RIGHT PANEL ----------------
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask Questions")
                
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about UTA courses...",
                    lines=2
                )
                
                submit_btn = gr.Button("Submit", variant="primary")
                
                response_output = gr.Textbox(
                    label="Response",
                    placeholder="Response will appear here...",
                    lines=15,
                    interactive=False
                )
                
                clear_btn = gr.Button("🗑 Clear")

        # ---------------- Event Logic ----------------

        # Init
        init_btn.click(
            fn=assistant.initialize_agent,
            outputs=init_status
        )

        # Submit query
        submit_btn.click(
            fn=assistant.process_query,
            inputs=query_input,
            outputs=response_output
        )
        
        query_input.submit(
            fn=assistant.process_query,
            inputs=query_input,
            outputs=response_output
        )

        # Clear
        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[query_input, response_output]
        )

    return demo, assistant


# ============================================================
#  MAIN FUNCTION
# ============================================================
def main():
    print("🚀 Starting UTA Course Q&A Agent...")
    print("📁 Checking for project_data.csv...")

    if os.path.exists("project_data.csv"):
        print("✅ Data file found.")
    else:
        print("⚠️ project_data.csv NOT FOUND!")
        print("Put it in the same folder as this script.")

    demo, assistant = create_gradio_interface()

    print("🌐 Launching Gradio...")
    print("➡️ Public URL will be generated since localhost is blocked.")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,      # Required for your Mac (fixes localhost error)
        show_error=True
    )


if __name__ == "__main__":
    main()
