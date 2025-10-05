import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { FileUpload } from "@/components/FileUpload";
import { Button } from "@/components/ui/button";
import { ArrowRight, Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

const Upload = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const navigate = useNavigate();

  const handleAnalyze = async () => {
    if (!file) {
      toast.error("Please upload a file first");
      return;
    }

    setIsAnalyzing(true);
    toast.loading("Analyzing your case file...");

    // Simulate file analysis
    setTimeout(() => {
      setIsAnalyzing(false);
      toast.success("Analysis complete! Found 5 similar cases");
      navigate("/similar-cases", { 
        state: { 
          uploadedFile: file.name,
          analysisComplete: true 
        } 
      });
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <section className="container mx-auto px-4 py-16">
        <div className="max-w-3xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12 slide-up">
            <h1 className="text-4xl font-bold text-foreground mb-4">
              Upload Legal Document
            </h1>
            <p className="text-lg text-muted-foreground">
              Upload your case file to find similar precedents and get AI-powered insights
            </p>
          </div>

          {/* Upload Area */}
          <div className="fade-in">
            <FileUpload onFileSelect={setFile} />

            {/* Analyze Button */}
            {file && (
              <div className="mt-8 flex justify-center scale-in">
                <Button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  variant="legal"
                  size="xl"
                  className="min-w-64"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      Analyze & Find Similar Cases
                      <ArrowRight className="h-5 w-5" />
                    </>
                  )}
                </Button>
              </div>
            )}
          </div>

          {/* Info Cards */}
          <div className="grid md:grid-cols-3 gap-6 mt-16">
            <InfoCard
              number="01"
              title="Upload Document"
              description="Submit your legal case file in PDF or DOCX format"
            />
            <InfoCard
              number="02"
              title="AI Analysis"
              description="Our AI analyzes your case and searches our database"
            />
            <InfoCard
              number="03"
              title="View Results"
              description="Get similar cases with citations and sentiment analysis"
            />
          </div>
        </div>
      </section>
    </div>
  );
};

const InfoCard = ({
  number,
  title,
  description,
}: {
  number: string;
  title: string;
  description: string;
}) => (
  <div className="p-6 bg-card border border-border rounded-xl shadow-soft hover:shadow-medium transition-smooth">
    <div className="accent-gradient text-accent-foreground w-12 h-12 rounded-lg flex items-center justify-center font-bold text-lg mb-4 shadow-gold">
      {number}
    </div>
    <h3 className="font-semibold text-foreground mb-2">{title}</h3>
    <p className="text-sm text-muted-foreground">{description}</p>
  </div>
);

export default Upload;
