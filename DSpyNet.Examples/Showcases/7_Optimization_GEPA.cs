// DSpyNet.Examples/Showcases/7_Optimization_GEPA.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using DSpyNet.DSPy.Core;
using DSpyNet.DSPy.Evaluation;
using DSpyNet.DSPy.Modules;
using DSpyNet.DSPy.Teleprompters;
using DSpyNet.Examples.Config;
using DSpyNet.Examples.Core;
using Microsoft.Extensions.Logging;

namespace DSpyNet.Examples.Showcases
{
    // --- SIGNATURES ---

    [DspInstruction("Read the support message and determine its urgency.")]
    public class UrgencySig : IDSpySignature
    {
        [DspInput(Prefix = "Message:")] public string Message { get; set; }
        [DspOutput(Prefix = "Urgency:", Description = "One of: low, medium, high")] public string Urgency { get; set; }
    }

    [DspInstruction("Read the support message and determine the sender's sentiment.")]
    public class SentimentSig : IDSpySignature
    {
        [DspInput(Prefix = "Message:")] public string Message { get; set; }
        [DspOutput(Prefix = "Sentiment:", Description = "One of: positive, neutral, negative")] public string Sentiment { get; set; }
    }

    [DspInstruction("Read the support message and list the facility-issue categories it mentions.")]
    public class CategoriesSig : IDSpySignature
    {
        [DspInput(Prefix = "Message:")] public string Message { get; set; }
        [DspOutput(Prefix = "Categories:", Description = "Comma-separated subset of: cleaning_services_scheduling, customer_feedback_and_complaints, emergency_repair_services, facility_management_issues, general_inquiries, quality_and_safety_concerns, routine_maintenance_requests, specialized_cleaning_services, sustainability_and_environmental_practices, training_and_support_requests")]
        public string Categories { get; set; }
    }

    // --- COMPOSITE MODULE ---

    public class FacilitySupportModule : Module
    {
        public Predict<UrgencySig> UrgencyPredict;
        public Predict<SentimentSig> SentimentPredict;
        public Predict<CategoriesSig> CategoriesPredict;

        public FacilitySupportModule(ILM lm, ILogger logger = null) : base(logger)
        {
            UrgencyPredict = new Predict<UrgencySig>(lm, logger);
            SentimentPredict = new Predict<SentimentSig>(lm, logger);
            CategoriesPredict = new Predict<CategoriesSig>(lm, logger);
        }

        public override async Task<object> InvokeAsync(object input)
        {
            var u = (Prediction)await UrgencyPredict.InvokeAsync(input);
            var s = (Prediction)await SentimentPredict.InvokeAsync(input);
            var c = (Prediction)await CategoriesPredict.InvokeAsync(input);
            return new Prediction(new Dictionary<string, object>
            {
                ["Urgency"] = u.Get<string>("Urgency") ?? "",
                ["Sentiment"] = s.Get<string>("Sentiment") ?? "",
                ["Categories"] = c.Get<string>("Categories") ?? ""
            });
        }

        public override Module DeepClone()
        {
            var clone = (FacilitySupportModule)MemberwiseClone();
            clone.UrgencyPredict = (Predict<UrgencySig>)UrgencyPredict.DeepClone();
            clone.SentimentPredict = (Predict<SentimentSig>)SentimentPredict.DeepClone();
            clone.CategoriesPredict = (Predict<CategoriesSig>)CategoriesPredict.DeepClone();
            return clone;
        }

        protected override void CopyStateFrom(Module source)
        {
            if (source is FacilitySupportModule s)
            {
                UrgencyPredict.State.Instruction = s.UrgencyPredict.State.Instruction;
                SentimentPredict.State.Instruction = s.SentimentPredict.State.Instruction;
                CategoriesPredict.State.Instruction = s.CategoriesPredict.State.Instruction;
            }
        }
    }

    // --- SHOWCASE ---

    public class OptimizationGepaExample : ExampleRunner
    {
        public override string Name => "Optimization (GEPA)";
        public override string Description => "Reflective Prompt Evolution on a multi-predictor Facility Support Analyzer.";

        public OptimizationGepaExample(AppConfig config, ILoggerFactory loggerFactory) : base(config, loggerFactory) { }

        public override async Task RunAsync()
        {
            PrintHeader("Running GEPA: Facility Support Analyzer");
            PrintInfo("Task: classify each facility-support email into Urgency / Sentiment / Categories.");
            PrintInfo("GEPA reflects on per-predictor failures and rewrites instructions.");

            var trainset = FacilitySupportData.Build();
            var student = new FacilitySupportModule(_lm, _logger);

            FeedbackMetric metric = (gold, pred, predName) =>
            {
                string gU = gold.Get<string>("Urgency"), pU = pred.Get<string>("Urgency");
                string gS = gold.Get<string>("Sentiment"), pS = pred.Get<string>("Sentiment");
                string gC = gold.Get<string>("Categories"), pC = pred.Get<string>("Categories");

                double sU = string.Equals(gU, pU, StringComparison.OrdinalIgnoreCase) ? 1 : 0;
                double sS = string.Equals(gS, pS, StringComparison.OrdinalIgnoreCase) ? 1 : 0;
                double sC = CategoryF1(gC, pC);
                double total = (sU + sS + sC) / 3.0;

                string fb = predName switch
                {
                    "UrgencyPredict" => $"Expected urgency '{gU}', got '{pU}'.",
                    "SentimentPredict" => $"Expected sentiment '{gS}', got '{pS}'.",
                    "CategoriesPredict" => $"Expected categories '{gC}', got '{pC}'.",
                    _ => $"Overall {total:F2} (U={sU} S={sS} C={sC:F2})"
                };
                return new ScoreFeedback(total, fb);
            };

            var evaluator = new Evaluator(_logger);
            var baseline = await evaluator.EvaluatePerExampleAsync(student, trainset, metric);
            PrintInfo($"Baseline avg score: {baseline.Average():F3}");

            var gepa = new GEPA<FacilitySupportModule>(
                reflectionLM: _lm,
                metric: metric,
                options: new GEPAOptions { Auto = GEPAAutoBudget.Light, Seed = 42 },
                logger: _logger);

            Console.WriteLine("\n[GEPA] Compiling...");
            var compiled = await gepa.CompileAsync(student, trainset);

            var after = await evaluator.EvaluatePerExampleAsync(compiled, trainset, metric);

            Console.WriteLine("\n=== GEPA Result ===");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"Optimized avg score: {after.Average():F3} (baseline {baseline.Average():F3})");
            Console.WriteLine($"  Urgency instr   : {compiled.UrgencyPredict.State.Instruction}");
            Console.WriteLine($"  Sentiment instr : {compiled.SentimentPredict.State.Instruction}");
            Console.WriteLine($"  Categories instr: {compiled.CategoriesPredict.State.Instruction}");
            Console.ResetColor();

            var demoMsg = "Generator room is flooding, water rising fast, please respond immediately";
            PrintInfo($"\nDemo input: '{demoMsg}'");
            var pred = (Prediction)await compiled.InvokeAsync(new { Message = demoMsg });
            PrintOutput("Urgency", pred.Get<string>("Urgency"));
            PrintOutput("Sentiment", pred.Get<string>("Sentiment"));
            PrintOutput("Categories", pred.Get<string>("Categories"));
        }

        private static double CategoryF1(string gold, string pred)
        {
            var g = SplitSet(gold);
            var p = SplitSet(pred);
            if (g.Count == 0 && p.Count == 0) return 1.0;
            if (g.Count == 0 || p.Count == 0) return 0.0;
            int tp = g.Intersect(p).Count();
            double prec = (double)tp / p.Count;
            double rec = (double)tp / g.Count;
            return prec + rec == 0 ? 0 : 2 * prec * rec / (prec + rec);
        }

        private static HashSet<string> SplitSet(string s) =>
            new HashSet<string>((s ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
                                StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Synthetic facility-support emails using the canonical DSPy GEPA tutorial schema.
    /// For the full 200-example benchmark dataset, see:
    ///   tutorial: https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/
    ///   raw data: https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/main/use-cases/facility-support-analyzer/dataset.json
    /// </summary>
    public static class FacilitySupportData
    {
        public static List<Example> Build() => new()
        {
            Example.From(
                ("Message", "AC unit on floor 3 stopped working, conference room is 90F, meeting in 1 hour"),
                ("Urgency", "high"), ("Sentiment", "negative"),
                ("Categories", "emergency_repair_services,facility_management_issues")),
            Example.From(
                ("Message", "Could we adjust the bi-weekly cleaning to Mondays instead of Wednesdays?"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "cleaning_services_scheduling")),
            Example.From(
                ("Message", "Broken window in lobby, glass on the floor, safety hazard"),
                ("Urgency", "high"), ("Sentiment", "negative"),
                ("Categories", "emergency_repair_services,quality_and_safety_concerns")),
            Example.From(
                ("Message", "Lightbulb out in stairwell B"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "routine_maintenance_requests")),
            Example.From(
                ("Message", "Elevator stuck between floors 2 and 3, person trapped inside"),
                ("Urgency", "high"), ("Sentiment", "negative"),
                ("Categories", "emergency_repair_services,quality_and_safety_concerns")),
            Example.From(
                ("Message", "Quarterly HVAC inspection due next month, please schedule"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "routine_maintenance_requests")),
            Example.From(
                ("Message", "Restroom on floor 4 has been leaking for a few days, getting worse"),
                ("Urgency", "medium"), ("Sentiment", "negative"),
                ("Categories", "facility_management_issues,routine_maintenance_requests")),
            Example.From(
                ("Message", "Thanks for fixing the boiler so quickly last week, the team really appreciated it"),
                ("Urgency", "low"), ("Sentiment", "positive"),
                ("Categories", "customer_feedback_and_complaints")),
            Example.From(
                ("Message", "Replace ceiling tiles in the east wing, minor staining, not urgent"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "routine_maintenance_requests")),
            Example.From(
                ("Message", "Fire alarm system test failed during audit, needs immediate attention"),
                ("Urgency", "high"), ("Sentiment", "negative"),
                ("Categories", "emergency_repair_services,quality_and_safety_concerns")),
            Example.From(
                ("Message", "Need post-construction deep clean for the renovated 5th floor before tenants move in next week"),
                ("Urgency", "medium"), ("Sentiment", "neutral"),
                ("Categories", "specialized_cleaning_services,cleaning_services_scheduling")),
            Example.From(
                ("Message", "Can your team run a training session for our staff on the new chemical handling procedures?"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "training_and_support_requests")),
            Example.From(
                ("Message", "Inquiring about your eco-friendly cleaning options and waste reduction practices"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "sustainability_and_environmental_practices,general_inquiries")),
            Example.From(
                ("Message", "Disappointed with last week's service quality, several areas were missed"),
                ("Urgency", "medium"), ("Sentiment", "negative"),
                ("Categories", "customer_feedback_and_complaints,quality_and_safety_concerns")),
            Example.From(
                ("Message", "What service plans do you offer for a 12-story office building?"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "general_inquiries")),
            Example.From(
                ("Message", "Power outage affecting servers in the IT room, backup batteries kicking in"),
                ("Urgency", "high"), ("Sentiment", "negative"),
                ("Categories", "emergency_repair_services,facility_management_issues")),
            Example.From(
                ("Message", "Strong gas smell in the basement maintenance area, evacuating staff now"),
                ("Urgency", "high"), ("Sentiment", "negative"),
                ("Categories", "emergency_repair_services,quality_and_safety_concerns")),
            Example.From(
                ("Message", "Water main burst is flooding the parking garage, vehicles at risk"),
                ("Urgency", "high"), ("Sentiment", "negative"),
                ("Categories", "emergency_repair_services,facility_management_issues")),
            Example.From(
                ("Message", "Loose railing on 6th floor balcony, immediate fall risk for anyone leaning on it"),
                ("Urgency", "high"), ("Sentiment", "negative"),
                ("Categories", "quality_and_safety_concerns,routine_maintenance_requests")),
            Example.From(
                ("Message", "Several office chairs in the bullpen are squeaky and uncomfortable"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "routine_maintenance_requests")),
            Example.From(
                ("Message", "Vending machine on 2nd floor is malfunctioning, taking money without dispensing"),
                ("Urgency", "low"), ("Sentiment", "negative"),
                ("Categories", "facility_management_issues")),
            Example.From(
                ("Message", "Lobby decor is looking dated, would like to discuss a refresh"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "general_inquiries")),
            Example.From(
                ("Message", "Heating in the north wing has been inconsistent for two weeks"),
                ("Urgency", "medium"), ("Sentiment", "negative"),
                ("Categories", "facility_management_issues,routine_maintenance_requests")),
            Example.From(
                ("Message", "Some staff are confused about the new building access process, can you clarify the steps?"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "training_and_support_requests")),
            Example.From(
                ("Message", "Carpet in conference room A is showing significant wear, time to plan a replacement"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "routine_maintenance_requests")),
            Example.From(
                ("Message", "We need additional recycling bins in the courtyard for our sustainability initiative"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "sustainability_and_environmental_practices")),
            Example.From(
                ("Message", "Coffee was spilled across the carpet during last night's event, needs specialized treatment"),
                ("Urgency", "medium"), ("Sentiment", "neutral"),
                ("Categories", "specialized_cleaning_services")),
            Example.From(
                ("Message", "Looking for a window-cleaning quote for our 14-story high-rise, exterior only"),
                ("Urgency", "medium"), ("Sentiment", "neutral"),
                ("Categories", "specialized_cleaning_services,cleaning_services_scheduling")),
            Example.From(
                ("Message", "Wanted to say thanks — your team's professionalism on the audit prep was outstanding"),
                ("Urgency", "low"), ("Sentiment", "positive"),
                ("Categories", "customer_feedback_and_complaints")),
            Example.From(
                ("Message", "Inquiring about your energy-saving practices and any LEED certifications you support"),
                ("Urgency", "low"), ("Sentiment", "neutral"),
                ("Categories", "sustainability_and_environmental_practices,general_inquiries"))
        };
    }
}
