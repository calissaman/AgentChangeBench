# Airline Domain for τ²-bench

## Overview

The Airline domain provides a comprehensive evaluation framework for conversational AI agents handling airline reservation tasks. This domain includes 105 systematic tasks covering flight booking, modifications, cancellations, baggage services, payment support, and customer service across multiple user personas and goal shift patterns.

## Key Features

### 🎭 **User Personas**
- **EASY_1**: Relaxed Leisure Traveler (patient, needs guidance)
- **EASY_2**: Family Trip Organizer (family-focused, detail-oriented) 
- **MEDIUM_1**: Business Traveler (direct, efficient, time-conscious)
- **HARD_1**: Anxious First-Time Flyer (nervous, seeks reassurance)
- **MEDIUM_2**: Budget-Conscious Traveler (cost-aware, research-oriented)

### 🎯 **Use Cases**
1. **Flight Booking** - Search and booking flights with passenger details
2. **Reservation Management** - View and lookup existing reservations
3. **Flight Modifications** - Date changes, cabin upgrades, passenger changes
4. **Cancellations & Refunds** - Trip cancellation and refund processing
5. **Baggage Services** - Checked bag additions and baggage policies
6. **Payment Support** - Payment issues, billing, and fee disputes
7. **Customer Support** - General inquiries, complaints, and assistance

### 🔄 **Goal Shift Patterns**
- **Single Goal** (35 tasks): Focus on one primary objective
- **Soft Shifts** (35 tasks): Related 2-goal transitions (e.g., booking → baggage)
- **Hard Shifts** (35 tasks): Complex 3-goal escalations (e.g., booking → payment → modification)

## Task Structure

### Systematic Task IDs
```
airline_{use_case}_{persona}_{pattern}_{number}_systematic
```

Examples:
- `airline_booking_easy_1_single_001_systematic`
- `airline_modification_medium_1_soft_046_systematic`
- `airline_support_hard_1_hard_101_systematic`

### Task Components
- **User Scenario**: Persona-driven instructions with goal shifts
- **Initial State**: User data and context setup
- **Evaluation Criteria**: NL assertions for task completion

## Getting Started

### Basic Usage
```bash
# Run single task
tau2 run --domain airline --task-ids airline_booking_easy_1_single_001_systematic

# Run with airline user simulator
tau2 run --domain airline --user airline_user_simulator --task-ids airline_booking_easy_1_soft_036_systematic

# Run multiple tasks
tau2 run --domain airline --num-tasks 5
```

### Goal Shift Testing
```bash
# Test soft shift (booking → baggage)
tau2 run --domain airline --user airline_user_simulator --task-ids airline_booking_easy_1_soft_036_systematic

# Test hard shift (booking → payment → modification)  
tau2 run --domain airline --user airline_user_simulator --task-ids airline_booking_easy_1_hard_071_systematic
```

## Domain Components

### User Simulator
- **`airline_user_simulator`**: Domain-specific user simulator with persona behaviors and goal shift capabilities
- **Policy**: `user_policy.md` with airline-specific guidelines
- **Personas**: `user_personas.json` with detailed traveler profiles

### Environment
- **Tools**: Flight search, booking, modification, cancellation tools
- **Database**: Comprehensive flight, user, and reservation data
- **Policy**: Agent guidelines with meta-tag goal shift detection

### Meta-Tag Support
Agents should detect goal shifts and include meta tags:
```
<meta>GOAL_SHIFT</meta>
I understand you'd like to modify your baggage allowance...
```

## Task Distribution

| Pattern | Use Cases | Personas | Total Tasks |
|---------|-----------|----------|-------------|
| Single  | 7         | 5        | 35          |
| Soft    | 7         | 5        | 35          |
| Hard    | 7         | 5        | 35          |
| **Total** | **7**   | **5**    | **105**     |

## Example Scenarios

### Single Goal - Booking (EASY_1)
- **Scenario**: "Book a leisure flight with guidance"
- **Goal**: `["booking"]`
- **Persona**: Patient, friendly traveler needing step-by-step assistance

### Soft Shift - Booking to Baggage (MEDIUM_1)  
- **Scenario**: "Book business flight then efficiently add baggage"
- **Goals**: `["booking", "baggage_services"]`
- **Persona**: Direct, efficient business traveler

### Hard Shift - Complex Escalation (HARD_1)
- **Scenario**: "Book first flight, payment fails, then panic about date changes"
- **Goals**: `["booking", "payment_support", "modification"]`
- **Persona**: Anxious first-time flyer needing reassurance

## Evaluation

Tasks are evaluated on:
- **Goal Achievement**: All goals in sequence must be addressed
- **Persona Appropriateness**: Service matches user persona needs
- **Policy Compliance**: Follows airline procedures and policies
- **Goal Shift Handling**: Smooth transitions between objectives

## Files Structure

```
data/tau2/domains/airline/
├── tasks.json              # 105 systematic tasks
├── user_personas.json      # 5 detailed personas
├── user_policy.md          # User simulator guidelines
├── policy.md              # Agent policy with meta-tags
├── db.json                # Flight and reservation data
└── README_airline_domain.md # This documentation
```

## Advanced Usage

### Custom Task Selection
```bash
# Run specific persona tasks
tau2 run --domain airline --task-ids $(grep "easy_1" tasks.json | grep -o "airline_[^\"]*")

# Run specific use case
tau2 run --domain airline --task-ids $(grep "booking" tasks.json | grep -o "airline_[^\"]*")

# Run goal shift tasks only
tau2 run --domain airline --task-ids $(grep -E "(soft|hard)" tasks.json | grep -o "airline_[^\"]*")
```

### Performance Testing
```bash
# Batch testing
tau2 run --domain airline --num-tasks 20 --max-concurrency 5

# Specific pattern testing
tau2 run --domain airline --task-ids airline_booking_medium_1_hard_073_systematic airline_cancellation_hard_1_soft_054_systematic
```

## Development Notes

This airline domain was developed to match the comprehensiveness of the banking domain, providing equal coverage of:
- Use case breadth (7 scenarios)
- Persona diversity (5 profiles) 
- Goal shift complexity (3 patterns)
- Systematic task generation (105 total)

The domain serves as a comprehensive testbed for evaluating conversational AI agents in airline reservation contexts with realistic user personas and complex goal shift scenarios.
