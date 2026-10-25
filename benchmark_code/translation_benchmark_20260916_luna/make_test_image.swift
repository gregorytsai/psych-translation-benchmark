import Foundation
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers

let output = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "image_clinical_note.png"
let width = 1800
let height = 720
let colorSpace = CGColorSpaceCreateDeviceRGB()
let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                              bytesPerRow: width * 4, space: colorSpace,
                              bitmapInfo: bitmapInfo) else {
    fatalError("could not create bitmap context")
}

context.setFillColor(CGColor(red: 0.98, green: 0.98, blue: 0.98, alpha: 1.0))
context.fill(CGRect(x: 0, y: 0, width: width, height: height))
context.setFillColor(CGColor(red: 0.08, green: 0.12, blue: 0.20, alpha: 1.0))
context.fill(CGRect(x: 0, y: height - 118, width: width, height: 118))

let lines = [
    ("PSYCHIATRY FOLLOW-UP", 54.0, 50.0, 612.0),
    ("After four weeks, panic frequency fell from 5 episodes to 2 per week.", 42.0, 50.0, 448.0),
    ("Continue sertraline 50 mg daily; monitor nausea and suicidal ideation.", 42.0, 50.0, 292.0)
]

for (text, size, x, y) in lines {
    let font = CTFontCreateWithName("Helvetica-Bold" as CFString, size, nil)
    let attrs: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(rawValue: kCTFontAttributeName as String): font,
        NSAttributedString.Key(rawValue: kCTForegroundColorAttributeName as String): CGColor(red: 0.06, green: 0.08, blue: 0.12, alpha: 1.0)
    ]
    let attributed = NSAttributedString(string: text, attributes: attrs)
    let line = CTLineCreateWithAttributedString(attributed)
    context.saveGState()
    context.textPosition = CGPoint(x: x, y: y)
    CTLineDraw(line, context)
    context.restoreGState()
}

guard let image = context.makeImage(), let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: output) as CFURL,
                                                                                           UTType.png.identifier as CFString, 1, nil) else {
    fatalError("could not create PNG destination")
}
CGImageDestinationAddImage(destination, image, nil)
guard CGImageDestinationFinalize(destination) else { fatalError("could not write PNG") }
