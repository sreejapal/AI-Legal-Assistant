import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { SimilarCaseCard } from "@/components/SimilarCaseCard";
import { PDFModal } from "@/components/PDFModal";
import { FileText, TrendingUp } from "lucide-react";
import { useLocation } from "react-router-dom";

// Mock data for similar cases
const mockCases = [
  {
    id: 1,
    title: "Smith v. Johnson Corp",
    court: "Supreme Court",
    year: 2022,
    confidence: 94,
    summary: "Landmark case regarding contractual obligations in commercial disputes with specific emphasis on force majeure clauses.",
    citations: [
      "Section 23(A) of the Contract Act as applied in commercial settings",
      "Precedent established in Wilson v. State (2019) regarding liability limitations",
    ],
    sentiment: "positive" as const,
  },
  {
    id: 2,
    title: "Doe v. Acme Industries",
    court: "Court of Appeals",
    year: 2021,
    confidence: 89,
    summary: "Case involving breach of contract and damages assessment in corporate litigation.",
    citations: [
      "Standard for calculating compensatory damages under Section 45(B)",
      "Reference to Miller v. Tech Corp (2018) on mitigation of damages",
    ],
    sentiment: "neutral" as const,
  },
  {
    id: 3,
    title: "State v. Williams LLC",
    court: "High Court",
    year: 2023,
    confidence: 85,
    summary: "Recent decision on corporate governance and fiduciary responsibilities in shareholder disputes.",
    citations: [
      "Corporate governance standards per Section 12 of Business Corporation Act",
      "Fiduciary duty principles established in Brown v. Capital Partners (2020)",
    ],
    sentiment: "negative" as const,
  },
  {
    id: 4,
    title: "Martinez v. Global Tech",
    court: "District Court",
    year: 2020,
    confidence: 82,
    summary: "Intellectual property dispute involving trade secrets and non-compete agreements.",
    citations: [
      "Trade secret protection under the Uniform Trade Secrets Act",
      "Reasonableness standard for non-compete clauses from Anderson v. Software Inc (2017)",
    ],
    sentiment: "positive" as const,
  },
  {
    id: 5,
    title: "Thompson v. Investment Partners",
    court: "Court of Appeals",
    year: 2019,
    confidence: 78,
    summary: "Securities litigation case addressing disclosure requirements and investor protection.",
    citations: [
      "Disclosure obligations under Section 10(b) of the Securities Exchange Act",
      "Standard of materiality from Reynolds v. Financial Corp (2016)",
    ],
    sentiment: "neutral" as const,
  },
];

const SimilarCases = () => {
  const [selectedCase, setSelectedCase] = useState<typeof mockCases[0] | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const location = useLocation();
  const uploadedFile = location.state?.uploadedFile || "case-document.pdf";

  const handleCaseClick = (caseData: typeof mockCases[0]) => {
    setSelectedCase(caseData);
    setModalOpen(true);
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <section className="container mx-auto px-4 py-16">
        {/* Header */}
        <div className="mb-12 slide-up">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 legal-gradient rounded-lg shadow-medium">
              <FileText className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Analyzed Document</p>
              <h2 className="text-xl font-semibold text-foreground">{uploadedFile}</h2>
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-foreground mb-2">
                Similar Cases Found
              </h1>
              <p className="text-lg text-muted-foreground">
                We found {mockCases.length} relevant precedents matching your case
              </p>
            </div>
            <div className="flex items-center gap-2 px-6 py-3 bg-accent/10 border border-accent/30 rounded-lg">
              <TrendingUp className="h-5 w-5 text-accent" />
              <div>
                <p className="text-sm text-muted-foreground">Avg. Match Score</p>
                <p className="text-2xl font-bold text-accent">
                  {Math.round(mockCases.reduce((acc, c) => acc + c.confidence, 0) / mockCases.length)}%
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Cases Grid */}
        <div className="grid md:grid-cols-2 gap-6 fade-in">
          {mockCases.map((caseData, index) => (
            <div key={caseData.id} style={{ animationDelay: `${index * 100}ms` }}>
              <SimilarCaseCard
                title={caseData.title}
                court={caseData.court}
                year={caseData.year}
                confidence={caseData.confidence}
                summary={caseData.summary}
                onClick={() => handleCaseClick(caseData)}
              />
            </div>
          ))}
        </div>
      </section>

      {/* PDF Modal */}
      {selectedCase && (
        <PDFModal
          open={modalOpen}
          onOpenChange={setModalOpen}
          caseTitle={selectedCase.title}
          citations={selectedCase.citations}
          sentiment={selectedCase.sentiment}
        />
      )}
    </div>
  );
};

export default SimilarCases;
