import Foundation
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers

let output = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "technical_marker.png"
let width = 1200
let height = 500
let colorSpace = CGColorSpaceCreateDeviceRGB()
let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                              bytesPerRow: width * 4, space: colorSpace,
                              bitmapInfo: bitmapInfo) else {
    fatalError("could not create bitmap context")
}

context.setFillColor(CGColor(red: 0.97, green: 0.97, blue: 0.97, alpha: 1.0))
context.fill(CGRect(x: 0, y: 0, width: width, height: height))
context.setStrokeColor(CGColor(red: 0.10, green: 0.20, blue: 0.35, alpha: 1.0))
context.setLineWidth(8)
context.stroke(CGRect(x: 20, y: 20, width: width - 40, height: height - 40))
let text = "TECHNICAL MARKER 20261023"
let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 64.0, nil)
let attrs: [NSAttributedString.Key: Any] = [
    NSAttributedString.Key(rawValue: kCTFontAttributeName as String): font,
    NSAttributedString.Key(rawValue: kCTForegroundColorAttributeName as String): CGColor(red: 0.08, green: 0.12, blue: 0.20, alpha: 1.0)
]
let attributed = NSAttributedString(string: text, attributes: attrs)
let line = CTLineCreateWithAttributedString(attributed)
context.textPosition = CGPoint(x: 120, y: 220)
CTLineDraw(line, context)

guard let image = context.makeImage(), let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: output) as CFURL,
                                                                                           UTType.png.identifier as CFString, 1, nil) else {
    fatalError("could not create PNG destination")
}
CGImageDestinationAddImage(destination, image, nil)
guard CGImageDestinationFinalize(destination) else { fatalError("could not write PNG") }
