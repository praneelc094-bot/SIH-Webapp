# DESIGN.md

## Overview
The "Figma Make App" is a productivity-focused dashboard designed to centralize tasks, analytics, and bookmarks. The visual language is clean and professional, utilizing a deep green color palette to convey stability and focus [inferred].

## Source Structure
- **Dashboard Layout**: The interface is structured as a customizable dashboard [inferred].
- **Content**: Includes sections for patient histories (7 records identified) [detected].
- **Navigation**: External links to the Figma community file are present [detected].

## Colors
| Role | Color | Usage |
| :--- | :--- | :--- |
| **Primary** | `#166534` | Main action buttons [detected] |
| **Secondary** | `#0D2E1A` | Text and secondary elements [detected] |
| **Background** | `#F0FAF4` | Page background [detected] |
| **Link** | `#4D7A5E` | Interactive text [detected] |

## Typography
- **Primary Font**: `DM Sans, system-ui, sans-serif` [detected].
- **Usage**: Applied as the general body font across the interface [detected].

## Layout
- **Color Scheme**: Light mode [detected].
- **Structure**: Dashboard-style layout with modular content blocks [inferred].

## Spacing/Shape/Depth
- **Border Radius**: `6px` is used consistently across components to provide a soft, modern aesthetic [detected].

## Components
- **Button Primary**: Background `#166534`, text `#FFFFFF`, radius `6px` [detected].
- **Button Secondary**: Background `#FFFFFF`, text `#4D7A5E`, border `1px solid #BBDECA`, radius `6px` [detected].

## Evidence Review
- **Color/Typography**: High confidence (0.9) based on semantic cleanup of branding signals [detected].
- **Components**: High confidence (0.95) for button styles [detected].
- **CTA Hierarchy**: No explicit CTA labels were detected; this area requires review during implementation [needs review].

## Do's And Don'ts
- **Do**: Use the detected color tokens (`#166534`, `#F0FAF4`) to maintain brand consistency.
- **Do**: Apply the `6px` border radius to all interactive components.
- **Don't**: Copy protected logos, proprietary UI, or source code wholesale [detected].
- **Don't**: Treat internal reasoning metadata as user-facing design truth [detected].

## Limitations
- This design is based on a snapshot of the provided URL and may not reflect real-time updates [detected].
- Certain noisy metadata (e.g., `scrapeId`, `creditsUsed`) was discarded during semantic cleanup [detected].

## Agent Usage Notes
- When implementing, prioritize the CSS variables provided:
  - `--utd-color-primary: #166534;`
  - `--utd-color-background: #F0FAF4;`
  - `--utd-color-textprimary: #0D2E1A;`
- Focus on building a responsive dashboard grid that accommodates the "customizable" nature of the app [inferred].
