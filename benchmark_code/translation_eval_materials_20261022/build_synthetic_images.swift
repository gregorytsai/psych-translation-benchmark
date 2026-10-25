import Foundation
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers

// These cards are intentionally synthetic translation/OCR stimuli. They are not
// scans, screenshots, or reproductions of any source document.

struct Card {
    let filename: String
    let title: String
    let subtitle: String
    let lines: [String]
    let accent: (CGFloat, CGFloat, CGFloat)
}

let outputDirectory = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "."
let width = 1600
let height = 900

let cards: [Card] = [
    Card(filename: "img01_synthetic_depression_note.png", title: "BEHAVIORAL HEALTH NOTE", subtitle: "SYNTHETIC EVALUATION IMAGE — NOT A CLINICAL RECORD", lines: [
        "Mood: low for most days; interest in hobbies has declined.",
        "Sleep is disrupted, and energy is reduced.",
        "Plan: review symptoms with a qualified health professional.",
        "Source cues: NIMH depression; WHO depressive disorder."
    ], accent: (0.16, 0.38, 0.66)),
    Card(filename: "img02_synthetic_panic_flow.png", title: "PANIC EPISODE FLOW", subtitle: "SYNTHETIC EVALUATION IMAGE — TEXT IS ORIGINAL", lines: [
        "unexpected surge → palpitations → shortness of breath",
        "check: substance, medical condition, or another disorder?",
        "track both episode frequency and functional recovery",
        "Source cue: adapted terminology from NIMH and OpenStax."
    ], accent: (0.75, 0.28, 0.24)),
    Card(filename: "img03_synthetic_ptsd_checklist.png", title: "TRAUMA RESPONSE CHECKLIST", subtitle: "SYNTHETIC EVALUATION IMAGE — NOT DIAGNOSTIC ADVICE", lines: [
        "□ intrusive memories or nightmares",
        "□ avoidance of reminders",
        "□ negative thoughts or feelings",
        "□ feeling on edge or easily startled",
        "Symptoms that persist and disrupt life need professional assessment.",
        "Source cue: VA National Center for PTSD."
    ], accent: (0.50, 0.32, 0.66)),
    Card(filename: "img04_synthetic_medication_label.png", title: "MEDICATION LABEL MOCKUP", subtitle: "SYNTHETIC EVALUATION IMAGE — NOT A PRESCRIPTION", lines: [
        "SERTRALINE — TAKE 50 mg ONCE DAILY",
        "Do not change the dose or stop suddenly without medical advice.",
        "Record nausea, sleep changes, and any urgent safety concern.",
        "Source cue: medication-safety wording is benchmark-authored."
    ], accent: (0.13, 0.52, 0.42)),
    Card(filename: "img05_synthetic_bipolar_timeline.png", title: "MOOD EPISODE TIMELINE", subtitle: "SYNTHETIC EVALUATION IMAGE — FICTIONAL EXAMPLE", lines: [
        "WEEK 1       depressive symptoms: low mood, low energy",
        "WEEK 2       sleep decreases; activity and speech increase",
        "WEEK 3       review lifetime pattern, family history, and medicines",
        "A single current symptom does not establish a diagnosis.",
        "Source cue: NIMH bipolar disorder."
    ], accent: (0.79, 0.47, 0.13)),
    Card(filename: "img06_synthetic_therapy_options.png", title: "PTSD TREATMENT OPTIONS", subtitle: "SYNTHETIC EVALUATION IMAGE — EDUCATIONAL MOCKUP", lines: [
        "CPT: reframe trauma-related thoughts",
        "PE: gradually approach safe memories and situations",
        "EMDR: process memories while attending to bilateral cues",
        "A clinician and patient choose an option together.",
        "Source cue: VA National Center for PTSD."
    ], accent: (0.24, 0.47, 0.44)),
    Card(filename: "img07_synthetic_suicide_prevention.png", title: "COMMUNITY SAFETY CARD", subtitle: "SYNTHETIC EVALUATION IMAGE — SUPPORT INFORMATION", lines: [
        "Protective factors include connection and access to care.",
        "Ask directly, listen without judgment, and connect to help.",
        "In the United States: call or text 988 for crisis support.",
        "Source cue: CDC Suicide Prevention; SAMHSA Find Help."
    ], accent: (0.61, 0.22, 0.29)),
    Card(filename: "img08_synthetic_mental_health_chart.png", title: "MENTAL HEALTH INFORMATION CARD", subtitle: "SYNTHETIC EVALUATION IMAGE — SOURCE-DERIVED TERMS", lines: [
        "Risk factors can include stress, trauma, isolation, and biology.",
        "Protective factors can include social support and practical care.",
        "Mental disorders are not character flaws or signs of weakness.",
        "Source cues: MedlinePlus; WHO mental health."
    ], accent: (0.28, 0.42, 0.58))
]

func color(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1.0) -> CGColor {
    return CGColor(red: r, green: g, blue: b, alpha: a)
}

func draw(_ text: String, at point: CGPoint, size: CGFloat, bold: Bool, foreground: CGColor, context: CGContext) {
    let name = bold ? "Helvetica-Bold" : "Helvetica"
    let font = CTFontCreateWithName(name as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(rawValue: kCTFontAttributeName as String): font,
        NSAttributedString.Key(rawValue: kCTForegroundColorAttributeName as String): foreground
    ]
    let attributed = NSAttributedString(string: text, attributes: attributes)
    let line = CTLineCreateWithAttributedString(attributed)
    context.saveGState()
    context.textPosition = point
    CTLineDraw(line, context)
    context.restoreGState()
}

for card in cards {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
    guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width * 4, space: colorSpace,
                                  bitmapInfo: bitmapInfo) else { fatalError("bitmap context") }
    context.setFillColor(color(0.97, 0.98, 0.99))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.setFillColor(color(card.accent.0, card.accent.1, card.accent.2))
    context.fill(CGRect(x: 0, y: height - 150, width: width, height: 150))
    context.setFillColor(color(1.0, 1.0, 1.0, 0.93))
    context.fill(CGRect(x: 90, y: 120, width: width - 180, height: height - 320))
    context.setStrokeColor(color(card.accent.0, card.accent.1, card.accent.2, 0.60))
    context.setLineWidth(4)
    context.stroke(CGRect(x: 90, y: 120, width: width - 180, height: height - 320))

    draw(card.title, at: CGPoint(x: 90, y: height - 78), size: 45, bold: true, foreground: color(1, 1, 1), context: context)
    draw(card.subtitle, at: CGPoint(x: 94, y: height - 120), size: 22, bold: false, foreground: color(1, 1, 1, 0.92), context: context)

    var y: CGFloat = 720
    for (index, line) in card.lines.enumerated() {
        let isSource = index == card.lines.count - 1
        draw(line, at: CGPoint(x: 150, y: y), size: isSource ? 22 : 35, bold: isSource, foreground: isSource ? color(0.20, 0.25, 0.32) : color(0.07, 0.11, 0.17), context: context)
        y -= isSource ? 72 : 86
    }
    context.setFillColor(color(card.accent.0, card.accent.1, card.accent.2, 0.11))
    context.fill(CGRect(x: 90, y: 64, width: width - 180, height: 42))
    draw("Synthetic image for translation/OCR evaluation • no real patient data • no endorsement", at: CGPoint(x: 120, y: 78), size: 20, bold: false, foreground: color(0.15, 0.18, 0.22), context: context)

    guard let image = context.makeImage(), let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: outputDirectory).appendingPathComponent(card.filename) as CFURL,
                                                                                               UTType.png.identifier as CFString, 1, nil) else { fatalError("PNG destination") }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { fatalError("PNG finalize") }
}
