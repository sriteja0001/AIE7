import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Forward the request to the Flask backend
    const response = await fetch('http://localhost:5001/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);

  } catch (error) {
    return NextResponse.json({ 
      error: 'Failed to connect to backend',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
} 