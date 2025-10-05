import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { ChatMessage } from "@/components/ChatMessage";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, Sparkles } from "lucide-react";
import { useNavigate } from "react-router-dom";
import heroImage from "@/assets/legal-hero.jpg";

interface Message {
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
}

const Home = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const navigate = useNavigate();

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      role: "user",
      content: input,
    };

    // Simulate AI response
    const aiResponse: Message = {
      role: "assistant",
      content: "I've analyzed your legal query. Based on similar precedents and legal frameworks, I recommend reviewing sections 23-A and 45-B of the relevant statute. Would you like me to provide more detailed analysis?",
      reasoning: "This response is based on pattern matching with 1,245 similar legal cases and cross-referencing with current statutory provisions.",
    };

    setMessages([...messages, userMessage, aiResponse]);
    setInput("");
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      
      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left: Illustration */}
          <div className="relative slide-up">
            <div className="absolute inset-0 legal-gradient opacity-20 blur-3xl rounded-full"></div>
            <img
              src={heroImage}
              alt="Legal AI Platform"
              className="relative rounded-2xl shadow-strong w-full"
            />
          </div>

          {/* Right: Chat Interface */}
          <div className="space-y-6 fade-in">
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Sparkles className="h-6 w-6 text-accent" />
                <Badge className="bg-accent/10 text-accent border-accent/30">
                  AI-Powered
                </Badge>
              </div>
              <h1 className="text-5xl font-bold text-foreground leading-tight">
                Your Legal
                <span className="block text-primary">Intelligence Partner</span>
              </h1>
              <p className="text-lg text-muted-foreground">
                Get instant AI-powered legal insights. Ask questions in any language,
                upload case files, and discover similar precedents.
              </p>
            </div>

            {/* Chat Box */}
            <div className="border border-border rounded-2xl bg-card shadow-medium overflow-hidden">
              {messages.length === 0 ? (
                <div className="p-8 text-center">
                  <div className="legal-gradient p-4 rounded-full w-16 h-16 mx-auto mb-4 shadow-medium">
                    <Sparkles className="h-8 w-8 text-primary-foreground" />
                  </div>
                  <h3 className="font-semibold text-foreground mb-2">
                    Start a Conversation
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    Ask me anything about legal matters or procedures
                  </p>
                </div>
              ) : (
                <div className="p-6 max-h-96 overflow-y-auto">
                  {messages.map((msg, idx) => (
                    <ChatMessage
                      key={idx}
                      role={msg.role}
                      content={msg.content}
                      reasoning={msg.reasoning}
                    />
                  ))}
                </div>
              )}

              {/* Input Area */}
              <div className="border-t border-border p-4 bg-muted/30">
                <div className="flex gap-3">
                  <Input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your legal question..."
                    className="flex-1 bg-background"
                  />
                  <Button onClick={handleSend} variant="legal" size="icon">
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* CTA Button */}
            <div className="flex gap-4">
              <Button
                onClick={() => navigate("/upload")}
                variant="gold"
                size="lg"
                className="flex-1"
              >
                Upload Case & Analyze
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border bg-muted/30 mt-20">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-sm text-muted-foreground">
              © 2025 LegalAI. AI-assisted legal research tool.
            </p>
            <div className="flex gap-6 text-sm text-muted-foreground">
              <a href="#" className="hover:text-foreground transition-smooth">
                Privacy Policy
              </a>
              <a href="#" className="hover:text-foreground transition-smooth">
                Terms of Service
              </a>
              <a href="#" className="hover:text-foreground transition-smooth">
                Contact
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

// Badge component for the hero section
const Badge = ({ children, className }: { children: React.ReactNode; className?: string }) => (
  <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border ${className}`}>
    {children}
  </span>
);

export default Home;
