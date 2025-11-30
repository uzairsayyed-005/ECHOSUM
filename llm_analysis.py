"""
LLM-based Meeting Analysis Module
Provides content analysis, summarization, and insight extraction for meeting transcripts.
"""

import os
import json
import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MeetingAnalysis:
    """Structure for comprehensive meeting analysis results"""
    executive_summary: str
    key_discussion_points: List[str]
    decisions_made: List[str]
    action_items: List[Dict[str, str]]  # [{"task": "", "owner": "", "deadline": ""}]
    follow_up_items: List[str]
    sentiment_analysis: str
    meeting_effectiveness_score: int  # 1-10
    generated_at: str

class LLMAnalyzer:
    """Meeting analysis using Large Language Models"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize LLM analyzer
        
        Args:
            provider: "openai", "anthropic", "ollama", or "groq"
            model: Model name (e.g., "gpt-3.5-turbo", "claude-3-sonnet", "llama3", "llama-3.1-70b-versatile")
            api_key: API key for cloud providers (not needed for Ollama)
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        # Initialize client based on provider
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.provider == "openai":
            try:
                import openai
                return openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        elif self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        elif self.provider == "groq":
            try:
                import groq
                return groq.Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("Groq package not installed. Run: pip install groq")
        
        elif self.provider == "ollama":
            try:
                import ollama
                return ollama
            except ImportError:
                raise ImportError("Ollama package not installed. Run: pip install ollama")
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call the LLM with the given prompt"""
        try:
            if self.provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=system_prompt or "You are a helpful meeting analysis assistant.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif self.provider == "groq":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            
            elif self.provider == "ollama":
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = self.client.generate(model=self.model, prompt=full_prompt)
                return response['response']
                
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def generate_executive_summary(self, transcript: str, speaker_stats: Dict) -> str:
        """Generate executive summary"""
        system_prompt = """You are an expert meeting analyst. Create a concise executive summary that captures the essence of the meeting in 2-3 paragraphs. Focus on the main purpose, key outcomes, and overall significance."""
        
        prompt = f"""
        Analyze this meeting transcript and create an executive summary.
        
        MEETING TRANSCRIPT:
        {transcript[:4000]}  # Limit for token constraints
        
        SPEAKER STATISTICS:
        {json.dumps(speaker_stats, indent=2)}
        
        Provide a 2-3 paragraph executive summary that includes:
        - Meeting purpose and context
        - Key outcomes and achievements
        - Overall significance and impact
        
        Keep it concise but comprehensive.
        """
        
        return self._call_llm(prompt, system_prompt)
    
    def extract_key_discussion_points(self, transcript: str) -> List[str]:
        """Extract main discussion topics and themes"""
        system_prompt = """You are an expert at identifying key discussion points in meetings. Extract the main topics, themes, and subjects that were discussed."""
        
        prompt = f"""
        Analyze this meeting transcript and identify the key discussion points.
        
        TRANSCRIPT:
        {transcript[:4000]}
        
        Extract 5-8 key discussion points. Each point should be:
        - A specific topic or theme that was discussed
        - Concise (1-2 sentences)
        - Meaningful and substantial
        
        Format as a numbered list:
        1. Point one
        2. Point two
        etc.
        """
        
        response = self._call_llm(prompt, system_prompt)
        
        # Parse numbered list into array
        points = []
        for line in response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                point = re.sub(r'^\d+\.\s*', '', line)
                if point:
                    points.append(point)
        
        return points
    
    def extract_decisions_made(self, transcript: str) -> List[str]:
        """Extract explicit decisions made during the meeting"""
        system_prompt = """You are an expert at identifying decisions made in meetings. Look for explicit decisions, resolutions, conclusions, and agreements reached by participants."""
        
        prompt = f"""
        Analyze this meeting transcript and identify all explicit decisions that were made.
        
        TRANSCRIPT:
        {transcript[:4000]}
        
        Extract decisions that were:
        - Explicitly stated or agreed upon
        - Clear resolutions or conclusions
        - Specific choices made between alternatives
        - Commitments or agreements reached
        
        Format as a numbered list:
        1. Decision one
        2. Decision two
        etc.
        
        If no clear decisions were made, respond with "No explicit decisions identified."
        """
        
        response = self._call_llm(prompt, system_prompt)
        
        if "no explicit decisions" in response.lower():
            return []
        
        # Parse numbered list
        decisions = []
        for line in response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                decision = re.sub(r'^\d+\.\s*', '', line)
                if decision:
                    decisions.append(decision)
        
        return decisions
    
    def extract_action_items(self, transcript: str) -> List[Dict[str, str]]:
        """Extract action items with owners and deadlines using enhanced patterns"""
        system_prompt = """You are an expert at identifying action items in meetings. Look for tasks, assignments, commitments, and follow-up actions. Pay special attention to phrases like "will do", "needs to", "should", "by [date]", "responsible for", "I'll", "we'll", "let's", and other commitment indicators."""
        
        prompt = f"""
        Analyze this meeting transcript and extract ALL action items, including implicit commitments and tasks.
        
        TRANSCRIPT:
        {transcript[:4000]}
        
        Look for action items in these forms:
        - Explicit assignments: "John will handle X"
        - Commitments: "I'll do Y by Friday"
        - Needs/shoulds: "We need to Z", "Someone should A"
        - Let's statements: "Let's schedule B"
        - Follow-ups: "I'll follow up with C"
        - Tasks with deadlines: "by tomorrow", "next week", "before the meeting"
        
        Format each action item as:
        ACTION: [Specific task or commitment] | OWNER: [Person responsible] | DEADLINE: [Timeframe if mentioned]
        
        Examples:
        ACTION: Schedule follow-up meeting with customer team | OWNER: Sarah | DEADLINE: This week
        ACTION: Review contract terms and provide feedback | OWNER: Mike | DEADLINE: By Friday
        ACTION: Update project documentation | OWNER: Team lead | DEADLINE: Next sprint
        
        Extract even vague commitments - it's better to capture too many than miss important items.
        """
        
        response = self._call_llm(prompt, system_prompt)
        
        if "no action items" in response.lower() and "action:" not in response.lower():
            return []
        
        # Enhanced parsing with multiple patterns
        action_items = []
        
        # Primary pattern: ACTION: ... | OWNER: ... | DEADLINE: ...
        primary_pattern = r'ACTION:\s*(.+?)\s*\|\s*OWNER:\s*(.+?)\s*\|\s*DEADLINE:\s*(.+?)(?=\n|ACTION:|$)'
        primary_matches = re.findall(primary_pattern, response, re.MULTILINE | re.IGNORECASE)
        
        for match in primary_matches:
            if len(match) == 3:
                action, owner, deadline = match
                action_items.append({
                    "task": action.strip(),
                    "owner": owner.strip(),
                    "deadline": deadline.strip()
                })
        
        # Fallback: Parse line by line for alternative formats
        if not action_items:
            items = response.split('---') if '---' in response else response.split('\n')
            
            for item in items:
                item = item.strip()
                if not item or item.lower().startswith('no action'):
                    continue
                    
                task_match = re.search(r'(?:TASK|ACTION):\s*(.+)', item, re.IGNORECASE)
                owner_match = re.search(r'OWNER:\s*(.+)', item, re.IGNORECASE)
                deadline_match = re.search(r'DEADLINE:\s*(.+)', item, re.IGNORECASE)
                
                if task_match:
                    action_item = {
                        "task": task_match.group(1).strip(),
                        "owner": owner_match.group(1).strip() if owner_match else "Not specified",
                        "deadline": deadline_match.group(1).strip() if deadline_match else "Not specified"
                    }
                    action_items.append(action_item)
        
        # Extract implicit action items from transcript
        implicit_actions = self._extract_implicit_actions(transcript)
        action_items.extend(implicit_actions)
        
        return action_items
        
    def _extract_implicit_actions(self, transcript: str) -> List[Dict[str, str]]:
        """Find action items from natural speech patterns in transcript"""
        actions = []
        
        # Patterns that indicate commitments/tasks
        commitment_patterns = [
            (r'(I|we|you|they|someone|somebody)\s+(?:will|gonna|going to)\s+(.+?)(?=\.|,|\n)', 'will_pattern'),
            (r'(I|we|you|they|someone|somebody)\s+(?:need to|should|have to)\s+(.+?)(?=\.|,|\n)', 'need_pattern'),
            (r'let\'s\s+(.+?)(?=\.|,|\n)', 'lets_pattern'),
            (r'(I|we)\'ll\s+(.+?)(?=\.|,|\n)', 'contraction_pattern'),
            (r'(\w+)\s+(?:is responsible for|will handle)\s+(.+?)(?=\.|,|\n)', 'responsibility_pattern'),
        ]
        
        for pattern, pattern_type in commitment_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            for match in matches:
                if pattern_type == 'lets_pattern':
                    actions.append({
                        "task": match.strip(),
                        "owner": "Team",
                        "deadline": "Not specified"
                    })
                elif pattern_type == 'responsibility_pattern' and len(match) == 2:
                    owner, task = match
                    actions.append({
                        "task": task.strip(),
                        "owner": owner.strip(),
                        "deadline": "Not specified"
                    })
                elif len(match) == 2:
                    owner, task = match
                    actions.append({
                        "task": task.strip(),
                        "owner": owner.strip(),
                        "deadline": "Not specified"
                    })
        
        # Limit to most relevant implicit actions (avoid noise)
        return actions[:3]  # Return top 3 implicit actions to avoid overwhelming
    
    def extract_follow_up_items(self, transcript: str) -> List[str]:
        """Extract items that need future discussion or follow-up"""
        system_prompt = """You are an expert at identifying follow-up items in meetings. Look for topics that were mentioned but not resolved, questions that need answers, or items deferred to future meetings."""
        
        prompt = f"""
        Analyze this meeting transcript and identify items that need follow-up or future discussion.
        
        TRANSCRIPT:
        {transcript[:4000]}
        
        Extract follow-up items such as:
        - Topics mentioned but not fully discussed
        - Questions raised but not answered
        - Issues deferred to future meetings
        - Items requiring additional research or information
        - Unresolved matters
        
        Format as a numbered list:
        1. Follow-up item one
        2. Follow-up item two
        etc.
        
        If no follow-up items were identified, respond with "No follow-up items identified."
        """
        
        response = self._call_llm(prompt, system_prompt)
        
        if "no follow-up items" in response.lower():
            return []
        
        # Parse numbered list
        follow_ups = []
        for line in response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                item = re.sub(r'^\d+\.\s*', '', line)
                if item:
                    follow_ups.append(item)
        
        return follow_ups
    
    def analyze_sentiment_and_effectiveness(self, transcript: str, speaker_stats: Dict) -> tuple[str, int]:
        """Analyze meeting sentiment and effectiveness"""
        system_prompt = """You are an expert at analyzing meeting dynamics, sentiment, and effectiveness."""
        
        prompt = f"""
        Analyze this meeting transcript for overall sentiment and effectiveness.
        
        TRANSCRIPT:
        {transcript[:3000]}
        
        SPEAKER STATISTICS:
        {json.dumps(speaker_stats, indent=2)}
        
        Provide:
        1. SENTIMENT: Overall meeting sentiment (positive, negative, neutral, mixed) with brief explanation
        2. EFFECTIVENESS: Rate meeting effectiveness on scale 1-10 with justification
        
        Format:
        SENTIMENT: [sentiment] - [brief explanation]
        EFFECTIVENESS: [score] - [justification]
        """
        
        response = self._call_llm(prompt, system_prompt)
        
        # Parse response
        sentiment = "Neutral - Unable to determine sentiment"
        effectiveness = 5
        
        sentiment_match = re.search(r'SENTIMENT:\s*(.+)', response, re.IGNORECASE)
        effectiveness_match = re.search(r'EFFECTIVENESS:\s*(\d+)', response, re.IGNORECASE)
        
        if sentiment_match:
            sentiment = sentiment_match.group(1).strip()
        if effectiveness_match:
            effectiveness = int(effectiveness_match.group(1))
        
        return sentiment, effectiveness
    
    def analyze_meeting(self, transcript_path: str, speaker_summary_path: str) -> MeetingAnalysis:
        """
        Perform comprehensive meeting analysis
        
        Args:
            transcript_path: Path to transcript.txt
            speaker_summary_path: Path to speaker_summary.txt
            
        Returns:
            MeetingAnalysis object with all insights
        """
        print("🤖 Starting LLM-based meeting analysis...")
        
        # Read transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        # Read speaker stats
        speaker_stats = {}
        try:
            with open(speaker_summary_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Parse speaker summary for basic stats
                speakers = re.findall(r"=== Speaker: (.+?) ===", content)
                speaker_stats = {"speakers": speakers, "summary": content[:500]}
        except Exception as e:
            print(f"Warning: Could not parse speaker summary: {e}")
        
        print("📝 Generating executive summary...")
        executive_summary = self.generate_executive_summary(transcript, speaker_stats)
        
        print("🎯 Extracting key discussion points...")
        key_points = self.extract_key_discussion_points(transcript)
        
        print("✅ Identifying decisions made...")
        decisions = self.extract_decisions_made(transcript)
        
        print("📋 Extracting action items...")
        action_items = self.extract_action_items(transcript)
        
        print("🔄 Identifying follow-up items...")
        follow_ups = self.extract_follow_up_items(transcript)
        
        print("😊 Analyzing sentiment and effectiveness...")
        sentiment, effectiveness = self.analyze_sentiment_and_effectiveness(transcript, speaker_stats)
        
        print("✨ Analysis complete!")
        
        return MeetingAnalysis(
            executive_summary=executive_summary,
            key_discussion_points=key_points,
            decisions_made=decisions,
            action_items=action_items,
            follow_up_items=follow_ups,
            sentiment_analysis=sentiment,
            meeting_effectiveness_score=effectiveness,
            generated_at=datetime.now().isoformat()
        )

def save_analysis_to_file(analysis: MeetingAnalysis, output_path: str = "meeting_analysis.txt"):
    """Save comprehensive analysis to file"""
    content = f"""
# Meeting Analysis Report
Generated: {analysis.generated_at}

## Executive Summary
{analysis.executive_summary}

## Key Discussion Points
"""
    
    for i, point in enumerate(analysis.key_discussion_points, 1):
        content += f"{i}. {point}\n"
    
    content += f"""
## Decisions Made
"""
    
    if analysis.decisions_made:
        for i, decision in enumerate(analysis.decisions_made, 1):
            content += f"{i}. {decision}\n"
    else:
        content += "No explicit decisions were identified.\n"
    
    content += f"""
## Action Items
"""
    
    if analysis.action_items:
        for i, item in enumerate(analysis.action_items, 1):
            content += f"{i}. **Task**: {item['task']}\n"
            content += f"   **Owner**: {item['owner']}\n"
            content += f"   **Deadline**: {item['deadline']}\n\n"
    else:
        content += "No action items were identified.\n"
    
    content += f"""
## Follow-up Items
"""
    
    if analysis.follow_up_items:
        for i, item in enumerate(analysis.follow_up_items, 1):
            content += f"{i}. {item}\n"
    else:
        content += "No follow-up items were identified.\n"
    
    content += f"""
## Meeting Assessment
**Sentiment**: {analysis.sentiment_analysis}
**Effectiveness Score**: {analysis.meeting_effectiveness_score}/10
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"📄 Analysis saved to {output_path}")

# Example usage function
def analyze_meeting_files(transcript_path: str = "transcript.txt", 
                         speaker_summary_path: str = "speaker_summary.txt",
                         provider: str = "openai",
                         model: str = "gpt-3.5-turbo"):
    """
    Convenience function to analyze meeting files
    
    Usage:
        # Using OpenAI (requires OPENAI_API_KEY environment variable)
        analyze_meeting_files(provider="openai", model="gpt-3.5-turbo")
        
        # Using Ollama (local)
        analyze_meeting_files(provider="ollama", model="llama3")
        
        # Using Anthropic (requires ANTHROPIC_API_KEY environment variable)
        analyze_meeting_files(provider="anthropic", model="claude-3-sonnet-20240229")
    """
    
    analyzer = LLMAnalyzer(provider=provider, model=model)
    analysis = analyzer.analyze_meeting(transcript_path, speaker_summary_path)
    save_analysis_to_file(analysis)
    
    return analysis

if __name__ == "__main__":
    # Example usage
    print("🚀 Meeting Analysis with LLM")
    print("Make sure you have:")
    print("1. transcript.txt file")
    print("2. speaker_summary.txt file") 
    print("3. Appropriate API key set as environment variable")
    print()
    
    # Run analysis with default settings (OpenAI)
    try:
        analysis = analyze_meeting_files()
        print("\n✅ Analysis completed successfully!")
        print(f"📄 Check 'meeting_analysis.txt' for detailed results")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Set OPENAI_API_KEY environment variable")
        print("- Or use: analyze_meeting_files(provider='ollama', model='llama3') for local LLM")