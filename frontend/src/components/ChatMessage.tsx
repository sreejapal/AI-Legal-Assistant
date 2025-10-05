import { Bot, User } from "lucide-react";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
}

export const ChatMessage = ({ role, content, reasoning }: ChatMessageProps) => {
  const isUser = role === "user";
  
  return (
    <div className={`flex gap-4 mb-6 fade-in ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="legal-gradient p-2 rounded-lg h-10 w-10 flex items-center justify-center shadow-medium">
          <Bot className="h-5 w-5 text-primary-foreground" />
        </div>
      )}
      
      <div className={`max-w-[75%] ${isUser ? 'order-first' : ''}`}>
        <div 
          className={`rounded-2xl px-5 py-3 shadow-soft ${
            isUser 
              ? 'bg-primary text-primary-foreground ml-auto' 
              : 'bg-card border border-border'
          }`}
        >
          <p className={`text-sm leading-relaxed ${isUser ? '' : 'text-foreground'}`}>
            {content}
          </p>
        </div>
        
        {reasoning && !isUser && (
          <details className="mt-3 p-3 bg-muted/50 rounded-lg border border-border">
            <summary className="cursor-pointer text-xs font-medium text-muted-foreground hover:text-foreground transition-smooth">
              View AI Reasoning
            </summary>
            <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
              {reasoning}
            </p>
          </details>
        )}
      </div>
      
      {isUser && (
        <div className="bg-accent p-2 rounded-lg h-10 w-10 flex items-center justify-center shadow-gold">
          <User className="h-5 w-5 text-accent-foreground" />
        </div>
      )}
    </div>
  );
};
